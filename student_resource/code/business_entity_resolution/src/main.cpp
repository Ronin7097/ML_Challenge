#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
using namespace std;

// No outside records or services are used. IDs are stored as their numeric suffix.
struct Rec { uint32_t id; string name, address; uint8_t country, source; };
struct Entry { uint64_t key; uint32_t row; bool operator<(const Entry& x) const { return key < x.key || (key == x.key && row < x.row); } };
struct Hit { uint32_t row; float retrieval; };
static vector<Rec> targets;
static vector<Entry> index_entries;
static constexpr int K=20;
using Features=array<float,K>;
struct Example { Features x; uint8_t y; uint32_t query, row; };

static uint8_t country_code(const string& s) {
    if(s=="US")return 1;if(s=="India")return 2;if(s=="France")return 3;
    static unordered_map<string,uint8_t> other;
    auto it=other.find(s);if(it!=other.end())return it->second;
    if(other.size()>=252){cerr<<"Too many distinct country labels\n";exit(1);}
    uint8_t code=uint8_t(4+other.size());other[s]=code;return code;
}
static uint32_t id_number(const string& s) { return (uint32_t)strtoul(s.c_str()+3,nullptr,10); }
static string id_text(uint8_t source,uint32_t id) { return "S"+to_string(source)+"-"+to_string(id); }
static bool parse_row(const string& line,Rec& r,uint8_t src) {
    size_t a=line.find('\t'),b=line.find('\t',a+1),c=line.find('\t',b+1);
    if(a==string::npos||b==string::npos||c==string::npos)return false;
    r.id=id_number(line.substr(0,a)); r.name=line.substr(a+1,b-a-1); r.address=line.substr(b+1,c-b-1);
    r.country=country_code(line.substr(c+1));r.source=src;return true;
}
static string lower(const string& s) {
    string z;z.reserve(s.size());
    for(unsigned char c:s)z.push_back(c>='A'&&c<='Z'?char(c+32):char(c));
    return z;
}
static string compact(const string&s);
static vector<string> words(const string& s) {
    vector<string> out;string t;
    for(unsigned char c:s) {
        bool keep=(c>='a'&&c<='z')||(c>='A'&&c<='Z')||(c>='0'&&c<='9')||c>=128;
        if(keep)t.push_back(c>='A'&&c<='Z'?char(c+32):char(c));
        else if(!t.empty()){out.push_back(t);t.clear();}
    }
    if(!t.empty())out.push_back(t);
    for(string& w:out)if(w.size()>1&&all_of(w.begin(),w.end(),[](char c){return c>='0'&&c<='9';})) {
        size_t p=w.find_first_not_of('0');w=p==string::npos?"0":w.substr(p);
    }
    sort(out.begin(),out.end());out.erase(unique(out.begin(),out.end()),out.end());return out;
}
static uint64_t hash_key(const string& s,uint8_t country,uint8_t type) {
    uint64_t h=1469598103934665603ULL;
    h=(h^country)*1099511628211ULL;h=(h^type)*1099511628211ULL;
    for(unsigned char c:s)h=(h^c)*1099511628211ULL;
    return h;
}
static vector<string> index_words(const string& s,int count) {
    auto v=words(s);
    v.erase(remove_if(v.begin(),v.end(),[](const string& w){return w.size()<3;}),v.end());
    stable_sort(v.begin(),v.end(),[](const string&a,const string&b){
        auto rank=[](const string&x){bool digit=all_of(x.begin(),x.end(),[](char c){return c>='0'&&c<='9';});return int(x.size())+(digit&&x.size()>=3?12:0);};
        return rank(a)>rank(b);
    });
    if((int)v.size()>count)v.resize(count);return v;
}
static vector<string> pair_words(const string& s) {
    auto v=words(s);v.erase(remove_if(v.begin(),v.end(),[](const string&w){return w.size()<2;}),v.end());
    stable_sort(v.begin(),v.end(),[](const string&a,const string&b){
        auto rank=[](const string&x){bool digit=all_of(x.begin(),x.end(),[](char c){return c>='0'&&c<='9';});return int(x.size())+(digit&&x.size()>=3?12:0);};
        return rank(a)>rank(b);
    });
    if(v.size()>5)v.resize(5);return v;
}
static string pair_text(const string&a,const string&b){return a<b?a+"|"+b:b+"|"+a;}
static void load_targets(const string& dir,const string& split) {
    // Build a sorted inverted index once per split. A posting contains a token key
    // and the row number of its Source 2 or Source 3 record.
    targets.clear();index_entries.clear();
    for(int src=2;src<=3;src++) {
        string path=dir+"/"+split+"_source"+to_string(src)+".tsv";
        ifstream in(path);if(!in){cerr<<"Cannot open "<<path<<"\n";exit(1);}string line;getline(in,line);
        while(getline(in,line)) {Rec r;if(!parse_row(line,r,src))continue;targets.push_back(std::move(r));}
        cerr<<"Loaded "<<path<<"; total targets "<<targets.size()<<"\n";
    }
    index_entries.reserve(targets.size()*18ULL);
    for(uint32_t i=0;i<targets.size();i++) {
        const Rec& r=targets[i];
        for(auto& w:index_words(r.name,4))index_entries.push_back({hash_key(w,r.country,1),i});
        for(auto& w:index_words(r.address,6))index_entries.push_back({hash_key(w,r.country,2),i});
        string exact=compact(lower(r.name));if(exact.size()>=4)index_entries.push_back({hash_key(exact,r.country,3),i});
        auto nw=pair_words(r.name),aw=pair_words(r.address);
        for(size_t j=1;j<nw.size();j++)index_entries.push_back({hash_key(pair_text(nw[j-1],nw[j]),r.country,4),i});
        for(size_t j=1;j<aw.size();j++)index_entries.push_back({hash_key(pair_text(aw[j-1],aw[j]),r.country,5),i});
    }
    sort(index_entries.begin(),index_entries.end());
    cerr<<"Indexed "<<index_entries.size()<<" postings\n";
}
static pair<size_t,size_t> posting(uint64_t key) {
    auto lo=lower_bound(index_entries.begin(),index_entries.end(),Entry{key,0});
    auto hi=upper_bound(lo,index_entries.end(),Entry{key,UINT32_MAX});
    return {size_t(lo-index_entries.begin()),size_t(hi-index_entries.begin())};
}
static vector<Hit> retrieve(const Rec& r,size_t limit=80) {
    // Search the rarest eligible token blocks first and rank unique target rows.
    // The returned list is exactly the candidate set scored at inference time.
    struct Block {size_t lo,hi;uint8_t type;};vector<Block> nb,ab,np,ap,exact;
    auto add=[&](const vector<string>& ws,uint8_t type,vector<Block>& blocks){
        for(const string& w:ws){if(w.size()<3)continue;auto p=posting(hash_key(w,r.country,type));
            if(p.second>p.first && p.second-p.first<=800)blocks.push_back({p.first,p.second,type});}
        sort(blocks.begin(),blocks.end(),[](const Block&a,const Block&b){return a.hi-a.lo<b.hi-b.lo;});
        if(blocks.size()>5)blocks.resize(5);
    };
    add(words(r.name),1,nb);add(words(r.address),2,ab);
    string normalized=compact(lower(r.name));if(normalized.size()>=4){auto p=posting(hash_key(normalized,r.country,3));
        if(p.second>p.first&&p.second-p.first<=800)exact.push_back({p.first,p.second,3});}
    auto pairs=[&](const vector<string>& ws,uint8_t type,vector<Block>& blocks){
        for(size_t i=0;i<ws.size();i++)for(size_t j=i+1;j<ws.size();j++){
            auto p=posting(hash_key(pair_text(ws[i],ws[j]),r.country,type));
            if(p.second>p.first&&p.second-p.first<=800)blocks.push_back({p.first,p.second,type});
        }
        sort(blocks.begin(),blocks.end(),[](const Block&a,const Block&b){return a.hi-a.lo<b.hi-b.lo;});
        if(blocks.size()>5)blocks.resize(5);
    };
    pairs(pair_words(r.name),4,np);pairs(pair_words(r.address),5,ap);
    unordered_map<uint32_t,float> score;score.reserve(500);
    size_t budget=0;
    auto scan=[&](const vector<Block>& blocks){for(const Block& b:blocks){size_t n=b.hi-b.lo;if(budget+n>1800)continue;budget+=n;
        float base=b.type==1?1.2f:b.type==2?1.0f:b.type==3?3.0f:b.type==4?2.0f:1.6f;
        float wt=base*(1.0f+log1pf(800.0f/n));
        for(size_t j=b.lo;j<b.hi;j++)score[index_entries[j].row]+=wt;
    }};
    scan(exact);scan(np);scan(ap);scan(nb);scan(ab);
    vector<Hit> hits;hits.reserve(score.size());for(auto& x:score)hits.push_back({x.first,x.second});
    if(hits.size()>limit){nth_element(hits.begin(),hits.begin()+limit,hits.end(),[](auto&a,auto&b){return a.retrieval>b.retrieval;});hits.resize(limit);}
    sort(hits.begin(),hits.end(),[](auto&a,auto&b){return a.retrieval>b.retrieval;});return hits;
}
static string compact(const string&s) {string o;for(unsigned char c:s)if((c>='a'&&c<='z')||(c>='0'&&c<='9')||c>=128)o.push_back(c);return o;}
static float dice3(const string&a,const string&b) {
    if(a.empty()||b.empty())return 0;if(a==b)return 1;
    if(a.size()<3||b.size()<3)return 0;
    unordered_map<uint32_t,int> m;m.reserve(a.size());
    auto key=[](const string&x,size_t i){return (uint32_t(uint8_t(x[i]))<<16)|(uint32_t(uint8_t(x[i+1]))<<8)|uint8_t(x[i+2]);};
    for(size_t i=0;i+3<=a.size();i++)m[key(a,i)]++;
    int shared=0;for(size_t i=0;i+3<=b.size();i++){auto it=m.find(key(b,i));if(it!=m.end()&&it->second>0){shared++;it->second--;}}
    return 2.0f*shared/(a.size()+b.size()-4);
}
static float token_overlap(const vector<string>&a,const vector<string>&b) {
    if(a.empty()||b.empty())return 0;int hit=0;size_t i=0,j=0;
    while(i<a.size()&&j<b.size()){if(a[i]==b[j]){hit++;i++;j++;}else if(a[i]<b[j])i++;else j++;}
    return float(hit)/float(min(a.size(),b.size()));
}
static float token_jaccard(const vector<string>&a,const vector<string>&b) {
    if(a.empty()||b.empty())return 0;int hit=0;size_t i=0,j=0;
    while(i<a.size()&&j<b.size()){if(a[i]==b[j]){hit++;i++;j++;}else if(a[i]<b[j])i++;else j++;}
    return float(hit)/float(a.size()+b.size()-hit);
}
static float number_overlap(const vector<string>&a,const vector<string>&b) {
    vector<string>x,y;for(auto&w:a)if(w.size()>=2&&isdigit((unsigned char)w[0]))x.push_back(w);
    for(auto&w:b)if(w.size()>=2&&isdigit((unsigned char)w[0]))y.push_back(w);
    return token_overlap(x,y);
}
static string first_number(const string&s) {
    string n;bool started=false;
    for(unsigned char c:s){if(c>='0'&&c<='9'){n.push_back(c);started=true;}
        else if(started)break;}
    if(n.empty())return n;size_t p=n.find_first_not_of('0');return p==string::npos?"0":n.substr(p);
}
static string core_name(const vector<string>&w) {
    static const unordered_set<string> legal={"llc","inc","pvt","private","ltd","limited","co","company","corp","corporation","lp","llp","plc","the"};
    string out;for(auto&x:w)if(!legal.count(x)){out+=x;out+='|';}return out;
}
static bool indic_script(const string&s){
    for(size_t i=0;i+2<s.size();i++){
        unsigned char a=s[i],b=s[i+1],c=s[i+2];
        if(a==0xE0&&b>=0xA4&&b<=0xB7&&c>=0x80&&c<=0xBF)return true;
    }
    return false;
}
static Features features(const Rec&a,const Rec&b,float retrieval) {
    // Pairwise name, address, numeric, and script features. Every value is in
    // [0,1] except the constant intercept feature.
    auto an=words(a.name),bn=words(b.name),aa=words(a.address),ba=words(b.address);
    string nac=compact(lower(a.name)),nbc=compact(lower(b.name));
    string aac=compact(lower(a.address)),bbc=compact(lower(b.address));
    float nd=dice3(nac,nbc),ad=dice3(aac,bbc);
    float no=token_overlap(an,bn),nj=token_jaccard(an,bn),ao=token_overlap(aa,ba),aj=token_jaccard(aa,ba);
    float nm=number_overlap(aa,ba);
    float contain=(nac.size()>5&&nbc.size()>5&&(nac.find(nbc)!=string::npos||nbc.find(nac)!=string::npos))?1:0;
    string na=first_number(a.address),nb=first_number(b.address);
    float ne=(!na.empty()&&na==nb),nc=(!na.empty()&&!nb.empty()&&na!=nb);
    float missing=b.address.empty(),sx=indic_script(a.name)!=indic_script(b.name);
    float core=!core_name(an).empty()&&core_name(an)==core_name(bn);
    float exact=!nac.empty()&&nac==nbc;
    return {1,nd,no,nj,ad,ao,aj,nm,contain,nd*ad,no*ao,max(nd,ad),min(nd,ad),
            exact,core,missing,ne,nc,sx,sx*ad};
}
static float sigmoid(float x){if(x>20)return 1;if(x< -20)return 0;return 1.0f/(1.0f+expf(-x));}
static float predict_score(const Features&x,const array<float,K>&w){float s=0;for(int i=0;i<K;i++)s+=x[i]*w[i];return sigmoid(s);}
static bool decide(const Features&x,float p,float threshold,int rules) {
    bool selected=p>=threshold;
    if((rules&1)&&x[14]>0.5f&&x[15]>0.5f&&x[1]>0.7f)selected=true;
    if((rules&2)&&x[18]>0.5f&&x[4]>=0.72f&&x[5]>=0.72f&&x[17]<0.5f)selected=true;
    if((rules&4)&&x[17]>0.5f&&x[1]<0.70f&&x[2]<0.70f)selected=false;
    return selected;
}
static vector<pair<uint32_t,vector<uint64_t>>> sample_truth(const string& path,int n) {
    ifstream in(path);string line;getline(in,line);mt19937 rng(2026);vector<pair<uint32_t,vector<uint64_t>>> out;
    uint64_t seen=0;while(getline(in,line)){size_t tab=line.find('\t');if(tab==string::npos)continue;
        uint32_t id=id_number(line);vector<uint64_t> matches;string rest=line.substr(tab+1);stringstream ss(rest);string m;
        while(getline(ss,m,','))if(m.size()>3)matches.push_back((uint64_t(m[1]-'0')<<32)|id_number(m));
        seen++;if(out.size()<(size_t)n)out.push_back({id,std::move(matches)});
        else {uniform_int_distribution<uint64_t> dis(0,seen-1);uint64_t j=dis(rng);if(j<(uint64_t)n)out[j]={id,std::move(matches)};}
    }
    return out;
}
static void diagnose(const string& base) {
    auto gt=sample_truth(base+"/dataset/train/train_ground_truth.tsv",3000);
    unordered_map<uint32_t,int> qmap;for(int i=0;i<(int)gt.size();i++)qmap[gt[i].first]=i;
    vector<Rec> queries(gt.size());ifstream in(base+"/dataset/train/train_source1.tsv");string line;getline(in,line);
    while(getline(in,line)){uint32_t id=id_number(line);auto it=qmap.find(id);if(it!=qmap.end())parse_row(line,queries[it->second],1);}
    load_targets(base+"/dataset/train","train");
    array<int,6> cutoff={20,50,80,150,300,1000};array<uint64_t,6> found{};uint64_t total=0;
    vector<pair<int,uint64_t>> missed;
    for(int qi=0;qi<(int)gt.size();qi++){
        auto hits=retrieve(queries[qi],1000);unordered_map<uint64_t,int> ranks;
        for(int j=0;j<(int)hits.size();j++){const Rec&b=targets[hits[j].row];ranks[(uint64_t(b.source)<<32)|b.id]=j+1;}
        for(uint64_t x:gt[qi].second){total++;int rank=ranks.count(x)?ranks[x]:100000;
            for(int j=0;j<6;j++)found[j]+=rank<=cutoff[j];
            if(rank>1000&&missed.size()<25)missed.push_back({qi,x});}
        if(qi%500==0)cerr<<"Diagnosed "<<qi<<"\n";
    }
    for(int j=0;j<6;j++)cerr<<"Recall@"<<cutoff[j]<<" "<<found[j]<<"/"<<total<<"="<<double(found[j])/total<<"\n";
    unordered_map<uint64_t,int> wanted;for(int i=0;i<(int)missed.size();i++)wanted[missed[i].second]=i;
    vector<const Rec*> target_miss(missed.size());for(const Rec&b:targets){uint64_t x=(uint64_t(b.source)<<32)|b.id;auto it=wanted.find(x);if(it!=wanted.end())target_miss[it->second]=&b;}
    for(int i=0;i<(int)missed.size();i++){const Rec&a=queries[missed[i].first];auto b=target_miss[i];
        cerr<<"MISS "<<id_text(1,a.id)<<" "<<a.name<<" | "<<a.address<<"\n";
        if(b)cerr<<"  "<<id_text(b->source,b->id)<<" "<<b->name<<" | "<<b->address<<"\n";}
}
static float macro_score(const vector<Example>& ex,const vector<int>& truth_count,const array<float,K>&w,float threshold,int from,int to,int rules=0) {
    vector<int> tp(to-from),fp(to-from);
    for(const Example&e:ex)if((int)e.query>=from&&(int)e.query<to&&decide(e.x,predict_score(e.x,w),threshold,rules)){if(e.y)tp[e.query-from]++;else fp[e.query-from]++;}
    double total=0;for(int i=from;i<to;i++){int t=truth_count[i],p=tp[i-from],f=fp[i-from];
        if(t==0&&p+f==0)total+=1;
        else if(p>0)total+=1.25*p/(1.25*p+0.25*(t-p)+f);
    }return total/(to-from);
}
static void train(const string& base,const string& modelpath) {
    // Reservoir-sample training entities, fit on 12k, and select the threshold
    // and rule flags on a disjoint 6k-entity macro-F0.5 validation sample.
    auto gt=sample_truth(base+"/dataset/train/train_ground_truth.tsv",18000);
    unordered_map<uint32_t,int> qmap;for(int i=0;i<(int)gt.size();i++)qmap[gt[i].first]=i;
    vector<Rec> queries(gt.size());ifstream in(base+"/dataset/train/train_source1.tsv");string line;getline(in,line);
    while(getline(in,line)){uint32_t id=id_number(line);auto it=qmap.find(id);if(it!=qmap.end())parse_row(line,queries[it->second],1);}
    load_targets(base+"/dataset/train","train");
    vector<Example> ex;vector<int> truth_count(gt.size());uint64_t retrieved=0,positives=0;
    for(int qi=0;qi<(int)gt.size();qi++){
        unordered_set<uint64_t> truth(gt[qi].second.begin(),gt[qi].second.end());truth_count[qi]=truth.size();positives+=truth.size();
        auto hits=retrieve(queries[qi]);for(auto& h:hits){const Rec& b=targets[h.row];bool y=truth.count((uint64_t(b.source)<<32)|b.id);
            retrieved+=y;ex.push_back({features(queries[qi],b,h.retrieval),(uint8_t)y,(uint32_t)qi,h.row});}
        if(qi%3000==0)cerr<<"Training queries "<<qi<<"/"<<gt.size()<<"\n";
    }
    cerr<<"Candidate recall "<<retrieved<<"/"<<positives<<" = "<<double(retrieved)/positives<<"; pairs "<<ex.size()<<"\n";
    const int split=12000;array<float,K>w{};w[0]=-3;
    // Stochastic logistic regression with stronger weight for positive pairs.
    mt19937 rng(87);vector<int> order;order.reserve(ex.size());for(int i=0;i<(int)ex.size();i++)if(ex[i].query<split)order.push_back(i);
    float overall_best=0,best_thr=0;array<float,K> best_w{};
    for(int epoch=0;epoch<8;epoch++){
        shuffle(order.begin(),order.end(),rng);float lr=0.025f/(1+epoch*.35f);
        for(int j:order){const auto&e=ex[j];float p=predict_score(e.x,w);float delta=lr*(e.y?3.0f:1.0f)*(e.y-p);
            for(int k=0;k<K;k++)w[k]+=delta*e.x[k];}
        float best=0,thr=0;for(int t=10;t<=98;t+=2){float x=t/100.0f,s=macro_score(ex,truth_count,w,x,split,gt.size());if(s>best){best=s;thr=x;}}
        cerr<<"Epoch "<<epoch+1<<" validation F0.5 "<<best<<" threshold "<<thr<<"\n";
        if(best>overall_best){overall_best=best;best_thr=thr;best_w=w;}
    }
    w=best_w;float best=0,thr=best_thr;int best_rules=0;
    for(int rules=0;rules<8;rules++)for(int t=65;t<=95;t+=2){float x=t/100.0f,s=macro_score(ex,truth_count,w,x,split,gt.size(),rules);
        if(s>best){best=s;thr=x;best_rules=rules;}}
    ofstream out(modelpath);out<<thr<<' '<<best_rules<<'\n';for(float x:w)out<<x<<' ';out<<'\n';
    cerr<<"Saved model "<<modelpath<<"; validation F0.5="<<best<<" threshold="<<thr<<" rules="<<best_rules<<"\n";
    int tp=0,fp=0,fn=0,shownfp=0,shownfn=0;
    for(const Example&e:ex)if((int)e.query>=split){float p=predict_score(e.x,w);bool pred=decide(e.x,p,thr,best_rules);
        if(e.y&&pred)tp++;else if(!e.y&&pred){fp++;if(shownfp++<12){const Rec&a=queries[e.query],&b=targets[e.row];cerr<<"FP "<<p<<" "<<a.name<<" | "<<a.address<<" <> "<<b.name<<" | "<<b.address<<"\n";}}
        else if(e.y){fn++;if(shownfn++<12){const Rec&a=queries[e.query],&b=targets[e.row];cerr<<"FN "<<p<<" "<<a.name<<" | "<<a.address<<" <> "<<b.name<<" | "<<b.address<<"\n";}}
    }
    cerr<<"Validation pair counts TP="<<tp<<" FP="<<fp<<" FN(retrieved)="<<fn<<"\n";
}
static void run(const string&base,const string&modelpath,const string&outdir) {
    // Stream Source 1 so the large test set does not require a second record pool.
    ifstream model(modelpath);if(!model){cerr<<"Cannot open model\n";exit(1);}float threshold;int rules;array<float,K>w;model>>threshold>>rules;for(auto&x:w)model>>x;
    load_targets(base+"/dataset/test","test");
    ifstream in(base+"/dataset/test/test_source1.tsv");string line;getline(in,line);
    ofstream matches(outdir+"/matching_results.tsv"),candidates(outdir+"/candidate_pairs.tsv");
    matches<<"source1_entity_id\tmatched_entity_ids\n";candidates<<"source1_entity_id\tcandidate_entity_ids\n";
    uint64_t n=0,pm=0,pc=0;while(getline(in,line)){Rec a;if(!parse_row(line,a,1))continue;auto hits=retrieve(a);
        string m,c;for(auto&h:hits){const Rec&b=targets[h.row];string id=id_text(b.source,b.id);if(!c.empty())c+=',';c+=id;pc++;
            Features x=features(a,b,h.retrieval);if(decide(x,predict_score(x,w),threshold,rules)){if(!m.empty())m+=',';m+=id;pm++;}}
        string sid=id_text(1,a.id);matches<<sid<<'\t'<<m<<'\n';candidates<<sid<<'\t'<<c<<'\n';n++;
        if(n%100000==0)cerr<<"Predicted "<<n<<" queries; "<<pc<<" candidates, "<<pm<<" matches\n";
    }
    cerr<<"Finished: "<<n<<" Source 1 rows, "<<pc<<" candidates, "<<pm<<" matches\n";
}
#ifndef RESOLVER_NO_MAIN
int main(int argc,char**argv){if(argc<3){cerr<<"Usage: resolver diagnose BASE | resolver train BASE MODEL | resolver predict BASE MODEL OUTPUT_DIR\n";return 2;}
    string mode=argv[1],base=argv[2];if(mode=="diagnose")diagnose(base);else if(mode=="train"&&argc>=4)train(base,argv[3]);else if(mode=="predict"&&argc>=5)run(base,argv[3],argv[4]);else return 2;}
#endif
