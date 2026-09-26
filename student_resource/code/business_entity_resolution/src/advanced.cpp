#define BOOST_NO_MAIN
#include "boost.cpp"
#include "advanced_features.h"
#include "reference_index.h"
#include "context_features.h"
#include <memory>

static void advanced_predict(const string&base,const string&pairpath,const string&contextpath,const string&outdir,bool use_reference,int limit,int max_queries,bool expected=false){
    ifstream context_header(contextpath);string version;int context_count;context_header>>version>>context_count;
    if(!context_header||version!="ERBOOST1"||(use_reference?(context_count!=FULL_K+CONTEXT_K+14&&context_count!=FULL_K+CONTEXT_K+REFERENCE_K):context_count!=FULL_K+CONTEXT_K))throw runtime_error("Incompatible context feature schema");
    BoostModel pair_model(pairpath,FULL_K),context_model(contextpath,context_count);
    unique_ptr<ReferenceIndex> references;if(use_reference)references=make_unique<ReferenceIndex>(base+"/dataset/test/test_source1.tsv");
    load_enhanced(base+"/dataset/test","test");filesystem::create_directories(outdir);
    ifstream in(base+"/dataset/test/test_source1.tsv");ofstream matching(outdir+"/matching_results.tsv"),candidates(outdir+"/candidate_pairs.tsv");
    if(!in||!matching||!candidates)throw runtime_error("Cannot open prediction files");string line;getline(in,line);
    matching<<"source1_entity_id\tmatched_entity_ids\n";candidates<<"source1_entity_id\tcandidate_entity_ids\n";size_t done=0,links=0;
    for(;;){vector<Rec> batch;while(batch.size()<128&&getline(in,line)){
        Rec r;if(!parse_row(line,r,1))throw runtime_error("Malformed test query");batch.push_back(std::move(r));if(max_queries>0&&done+batch.size()>=(size_t)max_queries)break;}
        if(batch.empty())break;struct Result{string matches,candidates;size_t links=0;};vector<Result> output(batch.size());
        parallel_queries(batch.size(),[&](size_t i){auto&q=batch[i];Prepared u(q);AdvancedPrepared au(q);auto hits=retrieve_enhanced(q,limit);
            vector<FullFeatures> x(hits.size());vector<double> scores(hits.size());vector<uint8_t> sources;vector<ReferenceFeatures> reference;
            if(references)reference.resize(hits.size());
            for(size_t j=0;j<hits.size();j++){auto&t=targets[hits[j].row];Prepared v(t,false);AdvancedPrepared av(t);
                auto f=extended_features(q,t,u,v,hits[j].retrieval,j);auto extra=advanced_features(au,av);copy(f.begin(),f.end(),x[j].begin());copy(extra.begin(),extra.end(),x[j].begin()+BOOST_K);
                sources.push_back(t.source);scores[j]=pair_model.raw(x[j]);if(references)reference[j]=references->features(q,t,x[j],scores[j],pair_model);}
            auto context=make_context(x,scores,sources,reference);auto&out=output[i];vector<double> final_scores;
            for(auto&features:context)final_scores.push_back(context_model.raw(features));auto decisions=choose_matches(final_scores,context_model.threshold,expected);
            for(size_t j=0;j<hits.size();j++){auto&t=targets[hits[j].row];auto id=id_text(t.source,t.id);if(!out.candidates.empty())out.candidates+=',';out.candidates+=id;
                if(decisions[j]){if(!out.matches.empty())out.matches+=',';out.matches+=id;out.links++;}}
        });
        for(size_t i=0;i<batch.size();i++){auto id=id_text(1,batch[i].id);matching<<id<<'\t'<<output[i].matches<<'\n';candidates<<id<<'\t'<<output[i].candidates<<'\n';links+=output[i].links;}
        done+=batch.size();if(done%1024==0)cerr<<"Predicted "<<done<<" queries; "<<links<<" links\n";if(max_queries>0&&done>=(size_t)max_queries)break;
    }
    if(!matching||!candidates)throw runtime_error("Prediction output write failure");cerr<<"Finished "<<done<<" queries; "<<links<<" links\n";
}

static void score_dynamic(const string&modelpath,const string&input,const string&output,int count){
    BoostModel model(modelpath,count);ifstream in(input,ios::binary);ofstream out(output,ios::binary);vector<float> x(count);
    if(!in||!out)throw runtime_error("Cannot open score files");
    while(in.read(reinterpret_cast<char*>(x.data()),count*sizeof(float))){double score=model.raw(x);out.write(reinterpret_cast<const char*>(&score),sizeof(score));}
    if(!in.eof()||in.gcount()!=0||!out)throw runtime_error("Incomplete score I/O");
}
static void verify_decisions(const string&input,const string&output,int count,double threshold,bool expected){
    vector<double> scores(count);ifstream in(input,ios::binary);if(!in.read(reinterpret_cast<char*>(scores.data()),count*sizeof(double)))throw runtime_error("Incomplete decision scores");
    auto decisions=choose_matches(scores,threshold,expected);ofstream out(output,ios::binary);for(bool decision:decisions){uint8_t value=decision;out.write(reinterpret_cast<const char*>(&value),1);}if(!out)throw runtime_error("Cannot write decisions");
}

static void export_context_test(const string&input,const string&scorespath,const string&sourcespath,const string&output,int count){
    vector<FullFeatures> x(count);vector<double> scores(count);vector<uint8_t> sources(count);
    ifstream data(input,ios::binary),values(scorespath,ios::binary),src(sourcespath,ios::binary);
    if(!data.read(reinterpret_cast<char*>(x.data()),count*sizeof(x[0]))||!values.read(reinterpret_cast<char*>(scores.data()),count*sizeof(scores[0]))||!src.read(reinterpret_cast<char*>(sources.data()),count))throw runtime_error("Incomplete context test inputs");
    auto context=make_context(x,scores,sources);ofstream out(output,ios::binary);for(auto&row:context)out.write(reinterpret_cast<const char*>(row.data()),row.size()*sizeof(float));
    if(!out)throw runtime_error("Cannot write context features");
}

static void reverse_export(const string&base,const string&input,const string&modelpath,const string&scorespath,const string&outpath,int qstart){
    BoostModel model(modelpath,BOOST_K+ADVANCED_K);ifstream meta(input+"/pairs.u32",ios::binary),scores(scorespath,ios::binary);
    if(!meta||!scores)throw runtime_error("Missing reverse inputs");
    unordered_set<uint64_t> wanted;array<uint32_t,5> row;double score;size_t total=0;
    while(meta.read(reinterpret_cast<char*>(row.data()),sizeof(row))){if(!scores.read(reinterpret_cast<char*>(&score),sizeof(score)))throw runtime_error("Missing pair score");
        if(int(row[0])>=qstart&&score>-8)wanted.insert((uint64_t(row[2])<<32)|row[1]);total++;}
    vector<uint32_t> query_ids;ifstream qfile(input+"/queries.tsv");string line;getline(qfile,line);
    while(getline(qfile,line)){size_t tab=line.find('\t');query_ids.push_back(id_number(line.substr(tab+1)));}
    unordered_map<uint64_t,Rec> selected;
    for(int src=2;src<=3;src++){ifstream in(base+"/dataset/train/train_source"+to_string(src)+".tsv");getline(in,line);while(getline(in,line)){
        uint64_t key=(uint64_t(src)<<32)|id_number(line);if(wanted.count(key)){Rec r;parse_row(line,r,src);selected.emplace(key,std::move(r));}}}
    if(selected.size()!=wanted.size())throw runtime_error("Missing requested target records");
    wanted.clear();cerr<<"Reverse comparisons for "<<selected.size()<<" unique targets\n";
    ReferenceIndex references(base+"/dataset/train/train_source1.tsv");
    meta.clear();meta.seekg(0);scores.clear();scores.seekg(0);ifstream data(input+"/features.f32",ios::binary);ofstream output(outpath,ios::binary);
    if(!data||!output)throw runtime_error("Cannot open reverse feature output");
    const size_t batchsize=2048;vector<array<uint32_t,5>> rows(batchsize);vector<double> values(batchsize);
    vector<array<float,BOOST_K+ADVANCED_K>> features(batchsize);vector<ReferenceFeatures> results(batchsize);size_t done=0,checked=0;
    while(done<total){size_t count=min(batchsize,total-done);
        meta.read(reinterpret_cast<char*>(rows.data()),count*sizeof(rows[0]));scores.read(reinterpret_cast<char*>(values.data()),count*sizeof(values[0]));data.read(reinterpret_cast<char*>(features.data()),count*sizeof(features[0]));
        if(!meta||!scores||!data)throw runtime_error("Partial reverse input batch");
        parallel_queries(count,[&](size_t i){results[i]={};auto&m=rows[i];if(int(m[0])<qstart||values[i]<=-8)return;
            auto&query=references.get(query_ids.at(m[0]));auto&target=selected.at((uint64_t(m[2])<<32)|m[1]);results[i]=references.features(query,target,features[i],values[i],model);});
        for(size_t i=0;i<count;i++)checked+=results[i][0]>0;output.write(reinterpret_cast<const char*>(results.data()),count*sizeof(results[0]));done+=count;
        if(done%262144==0)cerr<<"Reference features "<<done<<'/'<<total<<"; checked="<<checked<<'\n';
    }
    if(!output)throw runtime_error("Reverse output write failed");cerr<<"Reference features finished: "<<checked<<" pairs compared\n";
}

static void refresh_export(const string&base,const string&input,const string&output,int limit){
    filesystem::create_directories(output);vector<Rec> queries;ifstream qfile(input+"/queries.tsv");string line;getline(qfile,line);
    while(getline(qfile,line)){vector<string> fields;size_t last=0,pos;while((pos=line.find('\t',last))!=string::npos){fields.push_back(line.substr(last,pos-last));last=pos+1;}fields.push_back(line.substr(last));
        if(fields.size()!=10)throw runtime_error("Malformed query metadata");Rec r{};r.id=id_number(fields[1]);r.name=fields[8];r.address=fields[9];r.country=stoi(fields[4]);r.source=1;queries.push_back(std::move(r));}
    vector<unordered_set<uint64_t>> truth(queries.size());ifstream truthfile(input+"/truth.tsv");getline(truthfile,line);
    while(getline(truthfile,line)){size_t tab=line.find('\t');uint32_t qi=stoul(line.substr(0,tab));string id=line.substr(tab+1);truth.at(qi).insert((uint64_t(id[1]-'0')<<32)|id_number(id));}
    enable_sound_blocks();load_enhanced(base+"/dataset/train","train");
    ifstream oldmeta(input+"/pairs.u32",ios::binary),olddata(input+"/features.f32",ios::binary);
    ofstream meta(output+"/pairs.u32",ios::binary),data(output+"/features.f32",ios::binary),newtruth(output+"/truth.tsv");
    if(!oldmeta||!olddata||!meta||!data||!newtruth)throw runtime_error("Cannot open refreshed export files");newtruth<<"query\tmatched_entity_id\tretrieved\n";
    array<uint32_t,5> pending{};FullFeatures pending_x{};
    auto read=[&]{bool present=bool(oldmeta.read(reinterpret_cast<char*>(pending.data()),sizeof(pending)));if(present&&!olddata.read(reinterpret_cast<char*>(pending_x.data()),sizeof(pending_x)))throw runtime_error("Missing cached features");return present;};bool have=read();
    size_t done=0,found=0,total=0,reused=0;
    struct Row{unordered_map<uint64_t,FullFeatures> old;vector<FullFeatures>x;vector<array<uint32_t,5>>meta;string truth;size_t found=0,reused=0;};
    for(uint32_t start=0;start<queries.size();start+=128){vector<Row> batch(min<size_t>(128,queries.size()-start));
        for(size_t i=0;i<batch.size();i++){uint32_t qi=start+i;while(have&&pending[0]==qi){batch[i].old.emplace((uint64_t(pending[2])<<32)|pending[1],pending_x);have=read();}}
        parallel_queries(batch.size(),[&](size_t i){uint32_t qi=start+i;auto&q=queries[qi];auto&out=batch[i];Prepared u(q);AdvancedPrepared au(q);
            auto hits=retrieve_enhanced(q,limit);auto baseline=retrieve(q);unordered_set<uint32_t> old_rows;for(auto&h:baseline)old_rows.insert(h.row);unordered_set<uint64_t> seen;
            for(size_t j=0;j<hits.size();j++){auto&t=targets[hits[j].row];uint64_t key=(uint64_t(t.source)<<32)|t.id;FullFeatures x{};auto cached=out.old.find(key);
                if(cached!=out.old.end()){x=cached->second;x[58]=log1pf(hits[j].retrieval);x[59]=log1pf(j);out.reused++;}
                else {Prepared v(t,false);AdvancedPrepared av(t);auto f=extended_features(q,t,u,v,hits[j].retrieval,j);auto extra=advanced_features(au,av);copy(f.begin(),f.end(),x.begin());copy(extra.begin(),extra.end(),x.begin()+BOOST_K);}
                bool label=truth[qi].count(key);out.x.push_back(x);out.meta.push_back({qi,t.id,t.source,uint32_t(label),uint32_t(old_rows.count(hits[j].row))});if(label){out.found++;seen.insert(key);}}
            ostringstream gt;for(auto key:truth[qi])gt<<qi<<'\t'<<id_text(key>>32,uint32_t(key))<<'\t'<<seen.count(key)<<'\n';out.truth=gt.str();
        });
        for(size_t i=0;i<batch.size();i++){auto&r=batch[i];data.write(reinterpret_cast<const char*>(r.x.data()),r.x.size()*sizeof(FullFeatures));meta.write(reinterpret_cast<const char*>(r.meta.data()),r.meta.size()*sizeof(r.meta[0]));newtruth<<r.truth;found+=r.found;reused+=r.reused;total+=truth[start+i].size();done+=r.x.size();}
        if(start%1024==0)cerr<<"Refreshed "<<start+batch.size()<<'/'<<queries.size()<<" reused="<<reused<<'/'<<done<<'\n';
    }
    if(have||!data||!meta||!newtruth)throw runtime_error("Refresh export mismatch");filesystem::copy_file(input+"/queries.tsv",output+"/queries.tsv",filesystem::copy_options::overwrite_existing);
    filesystem::copy_file(input+"/schema.json",output+"/schema.json",filesystem::copy_options::overwrite_existing);
    cerr<<"Refreshed "<<done<<" pairs\n";
}

static void expand_export(const string&base,const string&cached,const string&plan,const string&output){
    vector<Rec> queries;ifstream qfile(plan+"/queries.tsv");string line;getline(qfile,line);
    while(getline(qfile,line)){vector<string> fields;size_t last=0,pos;while((pos=line.find('\t',last))!=string::npos){fields.push_back(line.substr(last,pos-last));last=pos+1;}fields.push_back(line.substr(last));
        if(fields.size()!=10)throw runtime_error("Malformed planned query");Rec r{};r.id=id_number(fields[1]);r.name=fields[8];r.address=fields[9];r.country=country_code(fields[4]);r.source=1;queries.push_back(std::move(r));}
    vector<unordered_set<uint64_t>> truth(queries.size());ifstream tf(plan+"/truth.tsv");getline(tf,line);
    while(getline(tf,line)){size_t tab=line.find('\t');uint32_t qi=stoul(line.substr(0,tab));string id=line.substr(tab+1);truth.at(qi).insert((uint64_t(id[1]-'0')<<32)|id_number(id));}
    vector<uint32_t> old_ids;ifstream oldq(cached+"/queries.tsv");getline(oldq,line);while(getline(oldq,line)){size_t tab=line.find('\t');old_ids.push_back(id_number(line.substr(tab+1)));}
    unordered_map<uint32_t,pair<size_t,size_t>> ranges;ifstream cached_meta(cached+"/pairs.u32",ios::binary);array<uint32_t,5> m{};size_t offset=0;
    while(cached_meta.read(reinterpret_cast<char*>(m.data()),sizeof(m))){auto&range=ranges[old_ids.at(m[0])];if(!range.second)range.first=offset;range.second++;offset++;}
    cached_meta.clear();ifstream cached_data(cached+"/features.f32",ios::binary);
    load_enhanced(base+"/dataset/train","train");
    ofstream data(output+"/features.f32",ios::binary),meta(output+"/pairs.u32",ios::binary),gt(output+"/truth.tsv");gt<<"query\tmatched_entity_id\tretrieved\n";
    if(!data||!meta||!gt||!cached_data)throw runtime_error("Cannot open expanded feature files");
    struct Row{vector<FullFeatures>x;vector<array<uint32_t,5>>meta;string truth;};size_t pairs=0;
    for(uint32_t start=0;start<queries.size();start+=128){vector<Row> batch(min<size_t>(128,queries.size()-start));
        for(size_t i=0;i<batch.size();i++){uint32_t qi=start+i;auto found=ranges.find(queries[qi].id);if(found==ranges.end())continue;
            auto range=found->second;auto&r=batch[i];r.x.resize(range.second);r.meta.resize(range.second);
            cached_data.seekg(range.first*sizeof(FullFeatures));cached_meta.seekg(range.first*sizeof(m));
            cached_data.read(reinterpret_cast<char*>(r.x.data()),r.x.size()*sizeof(FullFeatures));cached_meta.read(reinterpret_cast<char*>(r.meta.data()),r.meta.size()*sizeof(m));
            if(!cached_data||!cached_meta)throw runtime_error("Truncated cached query");for(auto&item:r.meta)item[0]=qi;
        }
        parallel_queries(batch.size(),[&](size_t i){uint32_t qi=start+i;auto&q=queries[qi];auto&out=batch[i];
            if(!ranges.count(q.id)){Prepared u(q);AdvancedPrepared au(q);auto hits=retrieve_enhanced(q,160);
                for(size_t j=0;j<hits.size();j++){auto&t=targets[hits[j].row];Prepared v(t,false);AdvancedPrepared av(t);auto basic=extended_features(q,t,u,v,hits[j].retrieval,j);auto extra=advanced_features(au,av);FullFeatures x{};
                    copy(basic.begin(),basic.end(),x.begin());copy(extra.begin(),extra.end(),x.begin()+BOOST_K);out.x.push_back(x);out.meta.push_back({qi,t.id,t.source,uint32_t(truth[qi].count((uint64_t(t.source)<<32)|t.id)),0});}}
            unordered_set<uint64_t> seen;for(auto&item:out.meta)if(item[3])seen.insert((uint64_t(item[2])<<32)|item[1]);ostringstream text;
            for(auto key:truth[qi])text<<qi<<'\t'<<id_text(key>>32,uint32_t(key))<<'\t'<<seen.count(key)<<'\n';out.truth=text.str();
        });
        for(auto&row:batch){data.write(reinterpret_cast<const char*>(row.x.data()),row.x.size()*sizeof(FullFeatures));meta.write(reinterpret_cast<const char*>(row.meta.data()),row.meta.size()*sizeof(m));gt<<row.truth;pairs+=row.x.size();}
        if(start%1024==0)cerr<<"Expanded "<<start+batch.size()<<'/'<<queries.size()<<" queries; "<<pairs<<" pairs\n";
    }
    if(!data||!meta||!gt)throw runtime_error("Expanded export write failure");
    filesystem::copy_file(plan+"/queries.tsv",output+"/queries.tsv",filesystem::copy_options::overwrite_existing);
    filesystem::copy_file(plan+"/schema.json",output+"/schema.json",filesystem::copy_options::overwrite_existing);
    cerr<<"Expanded export complete: "<<queries.size()<<" queries, "<<pairs<<" pairs\n";
}

static void augment(const string&base,const string&input,const string&output){
    filesystem::create_directories(output);
    vector<AdvancedPrepared> queries;ifstream qfile(input+"/queries.tsv");string line;getline(qfile,line);
    while(getline(qfile,line)){
        vector<string> fields;size_t last=0,pos;while((pos=line.find('\t',last))!=string::npos){fields.push_back(line.substr(last,pos-last));last=pos+1;}fields.push_back(line.substr(last));
        if(fields.size()!=10)throw runtime_error("Malformed query metadata");Rec r{};r.name=fields[8];r.address=fields[9];queries.emplace_back(r);
    }
    if(queries.empty())throw runtime_error("No queries to augment");
    vector<pair<uint64_t,uint32_t>> lookup;
    for(int src=2;src<=3;src++){
        ifstream file(base+"/dataset/train/train_source"+to_string(src)+".tsv");if(!file)throw runtime_error("Missing training source");getline(file,line);
        while(getline(file,line)){Rec r;if(!parse_row(line,r,src))throw runtime_error("Malformed target");lookup.push_back({(uint64_t(src)<<32)|r.id,uint32_t(targets.size())});targets.push_back(std::move(r));}
        cerr<<"Loaded "<<targets.size()<<" targets\n";
    }
    sort(lookup.begin(),lookup.end());ifstream meta(input+"/pairs.u32",ios::binary);ofstream out(output+"/advanced.f32",ios::binary);
    if(!meta||!out)throw runtime_error("Cannot open feature augmentation files");
    vector<array<uint32_t,5>> rows(8192);vector<AdvancedFeatures> features(8192);size_t done=0;
    for(;;){meta.read(reinterpret_cast<char*>(rows.data()),rows.size()*sizeof(rows[0]));size_t bytes=meta.gcount();
        if(bytes%sizeof(rows[0]))throw runtime_error("Partial metadata row");size_t count=bytes/sizeof(rows[0]);if(!count)break;
        parallel_queries(count,[&](size_t i){auto&m=rows[i];uint64_t key=(uint64_t(m[2])<<32)|m[1];auto it=lower_bound(lookup.begin(),lookup.end(),make_pair(key,uint32_t(0)));
            if(it==lookup.end()||it->first!=key||m[0]>=queries.size())throw runtime_error("Unknown pair ID");
            AdvancedPrepared target(targets[it->second]);features[i]=advanced_features(queries[m[0]],target);
        });
        out.write(reinterpret_cast<const char*>(features.data()),count*sizeof(features[0]));done+=count;
        if(done%262144==0)cerr<<"Augmented "<<done<<" pairs\n";
    }
    if(!out||(!meta.eof()&&meta.fail()))throw runtime_error("Feature augmentation I/O failure");
    ofstream names(output+"/advanced_names.json");names<<'[';for(size_t i=0;i<ADVANCED_NAMES.size();i++){if(i)names<<',';names<<'"'<<ADVANCED_NAMES[i]<<'"';}names<<"]\n";
    cerr<<"Augmented "<<done<<" total pairs\n";
}

#ifndef ADVANCED_NO_MAIN
int main(int argc,char**argv){try{
    if(argc==5&&string(argv[1])=="augment")augment(argv[2],argv[3],argv[4]);
    else if(argc==6&&string(argv[1])=="expand")expand_export(argv[2],argv[3],argv[4],argv[5]);
    else if(argc>=5&&string(argv[1])=="refresh")refresh_export(argv[2],argv[3],argv[4],argc>5?stoi(argv[5]):160);
    else if(argc==8&&string(argv[1])=="reverse")reverse_export(argv[2],argv[3],argv[4],argv[5],argv[6],stoi(argv[7]));
    else if(argc>=6&&string(argv[1])=="predict"){if(argc>9&&stoi(argv[9]))enable_sound_blocks();advanced_predict(argv[2],argv[3],argv[4],argv[5],argc>6?stoi(argv[6])!=0:false,argc>7?stoi(argv[7]):160,argc>8?stoi(argv[8]):0,argc>10?stoi(argv[10])!=0:false);}
    else if(argc==7&&string(argv[1])=="decide")verify_decisions(argv[2],argv[3],stoi(argv[4]),stod(argv[5]),stoi(argv[6])!=0);
    else if(argc==6&&string(argv[1])=="score")score_dynamic(argv[2],argv[3],argv[4],stoi(argv[5]));
    else if(argc==7&&string(argv[1])=="context")export_context_test(argv[2],argv[3],argv[4],argv[5],stoi(argv[6]));
    else if(argc==3&&string(argv[1])=="romanize")cout<<romanize(argv[2])<<'\n'<<join_words(sound_words(argv[2]))<<'\n';
    else {cerr<<"Usage: resolver_advanced augment BASE INPUT_EXPORT OUTPUT_EXPORT\n"
        <<"       resolver_advanced expand BASE CACHED_EXPORT PLAN OUTPUT\n"
        <<"       resolver_advanced refresh BASE INPUT_EXPORT OUTPUT [CANDIDATES=160]\n"
        <<"       resolver_advanced reverse BASE EXPORT PAIR_MODEL SCORES OUTPUT QUERY_BEGIN\n"
        <<"       resolver_advanced predict BASE PAIR_MODEL CONTEXT_MODEL OUTPUT [REFERENCE=0 CANDIDATES=160 MAX_QUERIES=0 PHONETIC=0 EXPECTED_F=0]\n"
        <<"       resolver_advanced score MODEL FEATURES SCORES FEATURE_COUNT\n";return 2;}
    return 0;
}catch(const exception&e){cerr<<"Error: "<<e.what()<<'\n';return 1;}}
#endif
