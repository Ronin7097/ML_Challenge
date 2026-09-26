// Compare each plausible target against other clean Source 1 records. This
// index is built entirely from unlabelled input fields, including at inference.
static const vector<string> REFERENCE_NAMES={"reference_checked","reference_best_logit","reference_second_logit",
    "reference_logit_margin","reference_better_count","reference_confident_count","reference_query_name_count","reference_target_name_count",
    "name_idf_fuzzy_query","name_idf_fuzzy_target","name_idf_exact_query","name_idf_exact_target","name_rarest_query_similarity","name_rarest_target_similarity",
    "first_digits_length_change","first_digits_length_ratio","first_digits_containment","first_digits_subsequence",
    "first_numeric_distance","first_numeric_one_apart","digit_omission_query","digit_omission_target",
    "unmatched_long_query_numbers","unmatched_long_target_numbers","unmatched_short_query_numbers","unmatched_short_target_numbers"};
static constexpr int REFERENCE_K=26;
using ReferenceFeatures=array<float,REFERENCE_K>;

struct ReferenceIndex {
    vector<Rec> records;vector<Entry> entries;vector<pair<uint32_t,uint32_t>> by_id;
    pair<size_t,size_t> find(uint64_t key)const {
        auto lo=lower_bound(entries.begin(),entries.end(),Entry{key,0});auto hi=upper_bound(lo,entries.end(),Entry{key,UINT32_MAX});
        return {size_t(lo-entries.begin()),size_t(hi-entries.begin())};
    }
    const Rec& get(uint32_t id)const {
        auto it=lower_bound(by_id.begin(),by_id.end(),make_pair(id,uint32_t(0)));
        if(it==by_id.end()||it->first!=id)throw runtime_error("Missing reference query");return records[it->second];
    }
    explicit ReferenceIndex(const string&path){
        ifstream in(path);if(!in)throw runtime_error("Missing reference source: "+path);string line;getline(in,line);
        while(getline(in,line)){Rec r;if(!parse_row(line,r,1))throw runtime_error("Malformed reference row");records.push_back(std::move(r));}
        entries.reserve(records.size()*24ULL);by_id.reserve(records.size());
        for(uint32_t i=0;i<records.size();i++){
            auto&r=records[i];by_id.push_back({r.id,i});auto cw=core_words(r.name),aw=address_words(r.address);
            auto add=[&](const string&s,uint8_t type){if(s.size()>=2)entries.push_back({hash_key(s,r.country,type),i});};
            add(compact(join_words(cw)),6);add(join_words(aw),7);for(auto&w:cw)if(w.size()>=3)add(w,12);
            for(auto&w:aw)if(w.size()>=3)add(w,9);for(auto&w:number_blocks(r.address,aw))add(w,10);
            for(size_t j=1;j<cw.size();j++)add(pair_text(cw[j-1],cw[j]),4);
        }
        sort(entries.begin(),entries.end());sort(by_id.begin(),by_id.end());cerr<<"Reference index: "<<records.size()<<" entities, "<<entries.size()<<" postings\n";
    }
    float name_count(const Rec&r)const{auto p=find(hash_key(compact(join_words(core_words(r.name))),r.country,6));return log1pf(p.second-p.first);}
    array<float,3> name_evidence(const vector<string>&a,const vector<string>&b,uint8_t country)const{
        double weights=0,fuzzy=0,exact=0,rarest=0,rare_similarity=0;
        for(auto&w:a){auto p=find(hash_key(w,country,12));double weight=min(12.,log1p(double(records.size())/(1+p.second-p.first)));float best=0;
            for(auto&v:b)best=max(best,w==v?1.f:edit_similarity(w,v));weights+=weight;fuzzy+=weight*best;exact+=weight*(best==1);
            if(weight>rarest){rarest=weight;rare_similarity=best;}}
        return {float(weights?fuzzy/weights:0),float(weights?exact/weights:0),float(rare_similarity)};
    }
    static bool subsequence(const string&shorter,const string&longer){if(shorter.empty()||shorter.size()>=longer.size())return false;size_t j=0;for(char c:longer)if(j<shorter.size()&&c==shorter[j])j++;return j==shorter.size();}
    static array<float,3> numeric_evidence(const vector<string>&a,const vector<string>&b){
        array<float,3> out{};for(auto&w:a)if(!binary_search(b.begin(),b.end(),w)){
            out[w.size()>=3?1:2]++;bool omission=false;for(auto&v:b)omission|=subsequence(v,w);out[0]+=omission;}
        return out;
    }
    vector<Hit> retrieve(const Rec&r,int limit=12)const {
        struct Block{size_t lo,hi;int type;};vector<Block> blocks;auto cw=core_words(r.name),aw=address_words(r.address);
        auto add=[&](const string&s,int type){if(s.size()<2)return;auto p=find(hash_key(s,r.country,type));if(p.second>p.first&&p.second-p.first<=12000)blocks.push_back({p.first,p.second,type});};
        add(compact(join_words(cw)),6);add(join_words(aw),7);for(auto&w:cw)if(w.size()>=3)add(w,12);
        for(auto&w:aw)if(w.size()>=3)add(w,9);for(auto&w:number_blocks(r.address,aw))add(w,10);
        for(size_t i=0;i<cw.size();i++)for(size_t j=i+1;j<cw.size();j++)add(pair_text(cw[i],cw[j]),4);
        sort(blocks.begin(),blocks.end(),[](const Block&a,const Block&b){return a.hi-a.lo!=b.hi-b.lo?a.hi-a.lo<b.hi-b.lo:a.lo<b.lo;});
        unordered_map<uint32_t,float> scores;size_t budget=0;int counts[13]={};
        for(auto&b:blocks){size_t n=b.hi-b.lo;if(budget+n>12000||counts[b.type]>=7)continue;budget+=n;counts[b.type]++;
            float weight=(b.type==6||b.type==7?4.f:b.type==10?3.f:b.type==4?2.f:1.f)*(1+log1pf(800.f/n));
            for(size_t j=b.lo;j<b.hi;j++)scores[entries[j].row]+=weight;
        }
        vector<Hit> hits;for(auto&item:scores)hits.push_back({item.first,item.second});
        auto better=[](const Hit&a,const Hit&b){return a.retrieval!=b.retrieval?a.retrieval>b.retrieval:a.row<b.row;};
        if(hits.size()>120){nth_element(hits.begin(),hits.begin()+120,hits.end(),better);hits.resize(120);}
        string cc=compact(join_words(cw));
        for(auto&h:hits){auto&v=records[h.row];auto nc=core_words(v.name),ac=address_words(v.address);float nd=dice3(cc,compact(join_words(nc))),ao=token_overlap(aw,ac);
            h.retrieval+=15*nd+10*token_overlap(cw,nc)+10*ao+20*nd*ao;}
        if(hits.size()>(size_t)limit){nth_element(hits.begin(),hits.begin()+limit,hits.end(),better);hits.resize(limit);}sort(hits.begin(),hits.end(),better);return hits;
    }
    ReferenceFeatures features(const Rec&q,const Rec&t,const array<float,BOOST_K+ADVANCED_K>&original,double score,const BoostModel&model)const{
        ReferenceFeatures result{};if(score<=-8)return result;result[0]=1;result[1]=result[2]=-30;result[6]=name_count(q);result[7]=name_count(t);
        auto qwords=core_words(romanize(q.name)),twords=core_words(romanize(t.name));auto qe=name_evidence(qwords,twords,q.country),te=name_evidence(twords,qwords,q.country);
        result[8]=qe[0];result[9]=te[0];result[10]=qe[1];result[11]=te[1];result[12]=qe[2];result[13]=te[2];
        string qn=first_number(q.address),tn=first_number(t.address);bool both=!qn.empty()&&!tn.empty();
        result[14]=both?float(int(tn.size())-int(qn.size())):0;result[15]=length_ratio(qn,tn);
        result[16]=both&&(qn.find(tn)!=string::npos||tn.find(qn)!=string::npos);result[17]=subsequence(qn,tn)||subsequence(tn,qn);
        double distance=both?abs(strtod(qn.c_str(),nullptr)-strtod(tn.c_str(),nullptr)):0;
        result[18]=log1p(min(distance,1e12));result[19]=both&&distance==1;
        auto qnums=digit_runs(q.address),tnums=digit_runs(t.address);auto qa=numeric_evidence(qnums,tnums),ta=numeric_evidence(tnums,qnums);
        result[20]=qa[0];result[21]=ta[0];result[22]=qa[1];result[23]=ta[1];result[24]=qa[2];result[25]=ta[2];
        Prepared v(t,false);AdvancedPrepared av(t);
        for(auto&h:retrieve(t)){
            auto&a=records[h.row];if(a.id==q.id)continue;Prepared u(a,false);AdvancedPrepared au(a);
            auto basic=extended_features(a,t,u,v,expm1f(original[58]),max(0,int(round(expm1f(original[59])))));
            // Retrieval evidence is held constant while comparing the text of
            // competing references, avoiding a change of score scale.
            basic[58]=original[58];basic[59]=original[59];basic[60]=original[60];
            auto extra=advanced_features(au,av);array<float,BOOST_K+ADVANCED_K>x{};copy(basic.begin(),basic.end(),x.begin());copy(extra.begin(),extra.end(),x.begin()+BOOST_K);
            double value=model.raw(x);result[4]+=value>score;result[5]+=value>2.2;
            if(value>result[1]){result[2]=result[1];result[1]=value;}else if(value>result[2])result[2]=value;
        }
        result[3]=score-result[1];return result;
    }
};
