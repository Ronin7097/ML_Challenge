// Reuse the original matcher for an auditable, paired baseline comparison.
#define RESOLVER_NO_MAIN
#include "main.cpp"
#include <filesystem>
#include <iomanip>
#include <stdexcept>
#include <atomic>
#include <thread>
#include <functional>

// Optional unlabelled blocking keys supplied by the advanced matcher.
static function<vector<pair<string,uint8_t>>(const Rec&)> additional_blocks;

template<class Function> static void parallel_queries(size_t count,Function work) {
    int workers=6;if(const char* value=getenv("ER_THREADS"))workers=max(1,min(16,atoi(value)));
    atomic<size_t> next{0};vector<thread> threads;vector<exception_ptr> errors(count);
    for(int t=0;t<workers;t++)threads.emplace_back([&]{for(;;){size_t i=next.fetch_add(1);if(i>=count)break;try{work(i);}catch(...){errors[i]=current_exception();}}});
    for(auto&t:threads)t.join();for(auto&e:errors)if(e)rethrow_exception(e);
}

static const vector<string> EXTRA_NAMES = {
    "core_exact", "core_dice", "core_edit", "core_overlap", "core_jaccard",
    "core_containment", "core_length_ratio", "name_length_ratio", "name_edit",
    "address_normalized_dice", "address_normalized_edit", "address_normalized_overlap",
    "address_normalized_jaccard", "address_exact", "address_length_ratio",
    "address_words_overlap", "address_words_dice", "numbers_overlap", "numbers_jaccard",
    "numbers_shared", "numbers_query_only", "numbers_target_only", "numbers_exact",
    "numbers_query_count", "numbers_target_count", "first_number_edit", "first_number_present",
    "name_fuzzy_query", "name_fuzzy_target", "name_unmatched_query", "name_unmatched_target",
    "core_query_tokens", "core_target_tokens", "address_query_tokens", "address_target_tokens",
    "query_missing_address", "query_indic", "target_indic", "retrieval_score", "retrieval_rank",
    "name_frequency", "query_name_length", "target_name_length", "same_source",
    "initials_equal", "initials_query_in_target", "initials_target_in_query",
    "core_bigram", "address_bigram", "street_number_conflict", "address_numberless_exact",
    "postal_shared", "postal_conflict", "name_prefix", "name_suffix", "address_containment"
};
static constexpr int BOOST_K = 76;
using BoostFeatures = array<float, BOOST_K>;

// This is character normalization, not a lookup of business identities.
static string fold_text(const string& s) {
    static const unordered_map<uint32_t,string> accents = {
        {192,"a"},{193,"a"},{194,"a"},{195,"a"},{196,"a"},{197,"a"},
        {224,"a"},{225,"a"},{226,"a"},{227,"a"},{228,"a"},{229,"a"},
        {199,"c"},{231,"c"},{200,"e"},{201,"e"},{202,"e"},{203,"e"},
        {232,"e"},{233,"e"},{234,"e"},{235,"e"},{204,"i"},{205,"i"},
        {206,"i"},{207,"i"},{236,"i"},{237,"i"},{238,"i"},{239,"i"},
        {209,"n"},{241,"n"},{210,"o"},{211,"o"},{212,"o"},{213,"o"},
        {214,"o"},{216,"o"},{242,"o"},{243,"o"},{244,"o"},{245,"o"},
        {246,"o"},{248,"o"},{217,"u"},{218,"u"},{219,"u"},{220,"u"},
        {249,"u"},{250,"u"},{251,"u"},{252,"u"},{221,"y"},{253,"y"},
        {255,"y"},{198,"ae"},{230,"ae"},{338,"oe"},{339,"oe"},{223,"ss"}
    };
    string out;
    for(size_t i=0;i<s.size();) {
        size_t start=i;uint32_t c=uint8_t(s[i++]);
        if(c>=192&&c<224&&i<s.size())c=((c&31)<<6)|(uint8_t(s[i++])&63);
        else if(c>=224&&c<240&&i+1<s.size()) {c=((c&15)<<12)|((uint8_t(s[i])&63)<<6)|(uint8_t(s[i+1])&63);i+=2;}
        auto it=accents.find(c);
        if(it!=accents.end())out+=it->second;
        else if(c<128)out+=char(c>='A'&&c<='Z'?c+32:c);
        else out+=s.substr(start,i-start);
    }
    return out;
}
static bool numeric(const string& s) {return !s.empty()&&all_of(s.begin(),s.end(),[](unsigned char c){return c>='0'&&c<='9';});}
static string join_words(const vector<string>& v) {string out;for(const auto&w:v){if(!out.empty())out+=' ';out+=w;}return out;}
static vector<string> core_words(const string& s) {
    static const unordered_set<string> stop = {
        "llc","inc","incorporated","pvt","private","ltd","limited","co","company",
        "corp","corporation","lp","llp","plc","the","and","www","com","org","net",
        "http","https","sri","shri","smt","sarl","sas","sa","eurl"
    };
    auto v=words(fold_text(s));vector<string> out;
    for(auto w:v) {
        if(stop.count(w))continue;
        if(!numeric(w))for(char&c:w){if(c=='0')c='o';else if(c=='1')c='l';else if(c=='5')c='s';else if(c=='6')c='g';}
        out.push_back(w);
    }
    sort(out.begin(),out.end());out.erase(unique(out.begin(),out.end()),out.end());return out;
}
static vector<string> address_words(const string& s) {
    static const unordered_map<string,string> aliases={
        {"road","rd"},{"street","st"},{"avenue","ave"},{"drive","dr"},{"boulevard","blvd"},
        {"lane","ln"},{"court","ct"},{"circle","cir"},{"place","pl"},{"highway","hwy"},
        {"parkway","pkwy"},{"terrace","ter"},{"apartment","apt"},{"apartments","apt"},
        {"appartment","apt"},{"suite","ste"},{"floor","fl"},{"building","bldg"},
        {"north","n"},{"south","s"},{"east","e"},{"west","w"},
        {"near","nr"},{"opposite","opp"},{"nagar","ngr"},{"saint","st"},
        {"maharashtra","mh"},{"gujarat","gj"},{"karnataka","ka"},{"telangana","ts"},
        {"rajasthan","rj"},{"punjab","pb"},{"haryana","hr"},{"kerala","kl"},
        {"महाराष्ट्र","mh"},{"తెలంగాణ","ts"},{"தமிழ்நாடு","tn"},{"दिल्ली","dl"},
        {"alabama","al"},{"alaska","ak"},{"arizona","az"},{"arkansas","ar"},
        {"california","ca"},{"colorado","co"},{"connecticut","ct"},{"delaware","de"},
        {"florida","fl"},{"georgia","ga"},{"hawaii","hi"},{"idaho","id"},
        {"illinois","il"},{"indiana","in"},{"iowa","ia"},{"kansas","ks"},
        {"kentucky","ky"},{"louisiana","la"},{"maine","me"},{"maryland","md"},
        {"massachusetts","ma"},{"michigan","mi"},{"minnesota","mn"},{"mississippi","ms"},
        {"missouri","mo"},{"montana","mt"},{"nebraska","ne"},{"nevada","nv"},
        {"ohio","oh"},{"oklahoma","ok"},{"oregon","or"},{"pennsylvania","pa"},
        {"tennessee","tn"},{"texas","tx"},{"utah","ut"},{"vermont","vt"},
        {"virginia","va"},{"washington","wa"},{"wisconsin","wi"},{"wyoming","wy"}
    };
    static const unordered_set<string> stop={"no","number","door","plot","flat","h","house","india","usa"};
    string normalized=fold_text(s);
    static const vector<pair<string,string>> phrases={
        {"west virginia","wv"},{"north carolina","nc"},{"south carolina","sc"},
        {"north dakota","nd"},{"south dakota","sd"},{"new york","ny"},
        {"new jersey","nj"},{"new mexico","nm"},{"new hampshire","nh"},{"rhode island","ri"},
        {"uttar pradesh","up"},{"madhya pradesh","mp"},{"tamil nadu","tn"},
        {"andhra pradesh","ap"},{"west bengal","wb"}
    };
    for(auto&p:phrases){size_t pos=normalized.find(p.first);if(pos!=string::npos)normalized.replace(pos,p.first.size(),p.second);}
    auto v=words(normalized);vector<string> out;
    for(auto w:v){if(stop.count(w))continue;auto it=aliases.find(w);if(it!=aliases.end())w=it->second;out.push_back(w);}
    sort(out.begin(),out.end());out.erase(unique(out.begin(),out.end()),out.end());return out;
}
static vector<string> numbers(const vector<string>& v) {vector<string> out;for(auto&w:v)if(numeric(w))out.push_back(w);return out;}
static float edit_similarity(const string& a,const string& b) {
    if(a.empty()||b.empty())return 0;if(a==b)return 1;
    // Bound work for unusually long address records.
    size_t n=min<size_t>(a.size(),192),m=min<size_t>(b.size(),192);
    array<int,193> prev{},cur{},prev2{};iota(prev.begin(),prev.begin()+m+1,0);
    for(size_t i=1;i<=n;i++) {cur[0]=i;for(size_t j=1;j<=m;j++){
        cur[j]=min({prev[j]+1,cur[j-1]+1,prev[j-1]+(a[i-1]!=b[j-1])});
        if(i>1&&j>1&&a[i-1]==b[j-2]&&a[i-2]==b[j-1])cur[j]=min(cur[j],prev2[j-2]+1);
    }prev2=prev;prev=cur;}
    return 1.0f-float(prev[m])/max(n,m);
}
static float dice2(const string&a,const string&b) {
    if(a.empty()||b.empty())return 0;if(a==b)return 1;if(min(a.size(),b.size())<2)return 0;
    vector<uint16_t>x,y;for(size_t i=1;i<a.size();i++)x.push_back((uint8_t(a[i-1])<<8)|uint8_t(a[i]));
    for(size_t i=1;i<b.size();i++)y.push_back((uint8_t(b[i-1])<<8)|uint8_t(b[i]));
    sort(x.begin(),x.end());sort(y.begin(),y.end());size_t i=0,j=0,hit=0;
    while(i<x.size()&&j<y.size()){if(x[i]==y[j]){i++;j++;hit++;}else if(x[i]<y[j])i++;else j++;}
    return 2.0f*hit/(x.size()+y.size());
}
static float length_ratio(const string&a,const string&b) {return max(a.size(),b.size())?float(min(a.size(),b.size()))/max(a.size(),b.size()):0;}
static float contained(const string&a,const string&b) {return min(a.size(),b.size())>=3&&(a.find(b)!=string::npos||b.find(a)!=string::npos);}
static pair<float,float> fuzzy_overlap(const vector<string>&a,const vector<string>&b) {
    float score=0,unmatched=0;for(const auto&w:a){float best=0;for(const auto&v:b){
        if(w==v){best=1;break;}if(min(w.size(),v.size())>=3&&length_ratio(w,v)>.5)best=max(best,edit_similarity(w,v));
    }score+=best;unmatched+=best<.75f;}return {a.empty()?0:score/a.size(),unmatched};
}
struct Prepared {
    vector<string> name,address,core,addr,nums,letters,postal;
    string nc,ac,cc,adc,al,first,initials;
    bool indic;float frequency;
    explicit Prepared(const Rec&r,bool frequency_lookup=true) {
        name=words(r.name);address=words(r.address);core=core_words(r.name);addr=address_words(r.address);
        nc=compact(lower(r.name));ac=compact(lower(r.address));cc=compact(join_words(core));
        adc=join_words(addr);nums=numbers(addr);first=first_number(r.address);indic=indic_script(r.name);
        for(const auto&w:addr)if(!numeric(w))letters.push_back(w);else if(w.size()==5||w.size()==6)postal.push_back(w);
        al=join_words(letters);for(const auto&w:core)if(!w.empty())initials+=w[0];
        auto p=frequency_lookup?posting(hash_key(cc,r.country,6)):make_pair(size_t(0),size_t(0));frequency=log1pf(p.second-p.first);
    }
};
static BoostFeatures extended_features(const Rec&a,const Rec&b,const Prepared&u,const Prepared&v,float retrieval,int rank) {
    BoostFeatures out{};int k=0;auto put=[&](float x){out.at(k++)=x;};
    float nd=dice3(u.nc,v.nc),ad=dice3(u.ac,v.ac),no=token_overlap(u.name,v.name),nj=token_jaccard(u.name,v.name);
    float ao=token_overlap(u.address,v.address),aj=token_jaccard(u.address,v.address);
    float ne=!u.first.empty()&&u.first==v.first,nc=!u.first.empty()&&!v.first.empty()&&u.first!=v.first,sx=u.indic!=v.indic;
    for(float x:Features{1,nd,no,nj,ad,ao,aj,number_overlap(u.address,v.address),
        float(u.nc.size()>5&&v.nc.size()>5&&contained(u.nc,v.nc)),nd*ad,no*ao,max(nd,ad),min(nd,ad),
        float(!u.nc.empty()&&u.nc==v.nc),float(!core_name(u.name).empty()&&core_name(u.name)==core_name(v.name)),
        float(b.address.empty()),ne,nc,sx,sx*ad})put(x);
    put(!u.cc.empty()&&u.cc==v.cc);put(dice3(u.cc,v.cc));put(edit_similarity(u.cc,v.cc));
    put(token_overlap(u.core,v.core));put(token_jaccard(u.core,v.core));put(contained(u.cc,v.cc));
    put(length_ratio(u.cc,v.cc));put(length_ratio(u.nc,v.nc));put(edit_similarity(u.nc,v.nc));
    put(dice3(u.adc,v.adc));put(edit_similarity(u.adc,v.adc));put(token_overlap(u.addr,v.addr));put(token_jaccard(u.addr,v.addr));
    put(!u.adc.empty()&&u.adc==v.adc);put(length_ratio(u.adc,v.adc));put(token_overlap(u.letters,v.letters));put(dice3(u.al,v.al));
    put(token_overlap(u.nums,v.nums));put(token_jaccard(u.nums,v.nums));
    int shared=0;for(auto&w:u.nums)shared+=binary_search(v.nums.begin(),v.nums.end(),w);
    put(shared);put(u.nums.size()-shared);put(v.nums.size()-shared);put(!u.nums.empty()&&u.nums==v.nums);
    put(u.nums.size());put(v.nums.size());put(edit_similarity(u.first,v.first));put(!u.first.empty()&&!v.first.empty());
    auto fq=fuzzy_overlap(u.core,v.core),ft=fuzzy_overlap(v.core,u.core);put(fq.first);put(ft.first);put(fq.second);put(ft.second);
    put(u.core.size());put(v.core.size());put(u.addr.size());put(v.addr.size());put(a.address.empty());put(u.indic);put(v.indic);
    put(log1pf(retrieval));put(log1pf(rank));put(u.frequency);put(u.nc.size());put(v.nc.size());put(b.source==2);
    put(!u.initials.empty()&&u.initials==v.initials);put(u.initials.size()>1&&v.cc.find(u.initials)!=string::npos);
    put(v.initials.size()>1&&u.cc.find(v.initials)!=string::npos);put(dice2(u.cc,v.cc));put(dice2(u.adc,v.adc));
    put(nc&&token_overlap(u.letters,v.letters)>.8f);put(!u.al.empty()&&u.al==v.al);
    put(token_overlap(u.postal,v.postal));put(!u.postal.empty()&&!v.postal.empty()&&!token_overlap(u.postal,v.postal));
    size_t pref=0,suff=0;while(pref<min(u.cc.size(),v.cc.size())&&u.cc[pref]==v.cc[pref])pref++;
    while(suff<min(u.cc.size(),v.cc.size())&&u.cc[u.cc.size()-1-suff]==v.cc[v.cc.size()-1-suff])suff++;
    put(min(u.cc.size(),v.cc.size())?float(pref)/min(u.cc.size(),v.cc.size()):0);
    put(min(u.cc.size(),v.cc.size())?float(suff)/min(u.cc.size(),v.cc.size()):0);put(contained(u.adc,v.adc));
    if(k!=BOOST_K)throw runtime_error("feature schema mismatch");return out;
}
static string consonants(const string& s) {string out;for(char c:s)if(string("aeiou").find(c)==string::npos)out+=c;return out;}
static string loose_core(const vector<string>&v) {
    static const unordered_set<string> generic={"enterprises","enterprise","center","services","service","trading","partners","store"};
    string out;for(auto&w:v)if(!generic.count(w))out+=w;return out;
}
static vector<string> number_blocks(const string&address,const vector<string>&aw) {
    vector<string> ns=numbers(aw),ws;
    string first=first_number(address);if(!first.empty()&&!binary_search(ns.begin(),ns.end(),first))ns.push_back(first);
    stable_sort(ns.begin(),ns.end(),[&](const string&a,const string&b){if((a==first)!=(b==first))return a==first;return a.size()>b.size();});
    if(ns.size()>3)ns.resize(3);
    for(auto&w:aw)if(w.size()>=3&&!numeric(w))ws.push_back(w);
    stable_sort(ws.begin(),ws.end(),[](const string&a,const string&b){return a.size()>b.size();});if(ws.size()>4)ws.resize(4);
    vector<string> out;for(auto&n:ns)for(auto&w:ws)out.push_back(n+"|"+w);return out;
}
static void load_enhanced(const string&dir,const string&split,int first_source=2,int last_source=3) {
    // Keep the baseline index unchanged and add two normalized whole-field keys.
    // This makes the baseline and the new retrieval directly comparable.
    targets.clear();index_entries.clear();
    for(int src=first_source;src<=last_source;src++){
        string path=dir+"/"+split+"_source"+to_string(src)+".tsv";ifstream in(path);
        if(!in)throw runtime_error("Cannot open "+path);string line;getline(in,line);
        while(getline(in,line)){Rec r;if(!parse_row(line,r,src))throw runtime_error("Malformed record in "+path);targets.push_back(std::move(r));}
        cerr<<"Loaded "<<targets.size()<<" targets\n";
    }
    index_entries.reserve(targets.size()*34ULL);
    for(uint32_t i=0;i<targets.size();i++){
        const auto&r=targets[i];auto add=[&](const string& s,uint8_t type){if(s.size()>=3)index_entries.push_back({hash_key(s,r.country,type),i});};
        auto indexed_names=index_words(r.name,4),indexed_addresses=index_words(r.address,6);
        for(auto&w:indexed_names)add(w,1);for(auto&w:indexed_addresses)add(w,2);
        add(compact(lower(r.name)),3);auto nw=pair_words(r.name),aw=pair_words(r.address);
        for(size_t j=1;j<nw.size();j++)add(pair_text(nw[j-1],nw[j]),4);
        for(size_t j=1;j<aw.size();j++)add(pair_text(aw[j-1],aw[j]),5);
        auto cw=core_words(r.name),awords=address_words(r.address);string core=compact(join_words(cw));add(core,6);
        add(join_words(awords),7);add(consonants(core),8);
        for(auto&w:awords)if(w.size()>=3&&find(indexed_addresses.begin(),indexed_addresses.end(),w)==indexed_addresses.end())add(w,9);
        for(auto&w:number_blocks(r.address,awords))add(w,10);add(loose_core(cw),11);
        for(auto&w:cw)if(w.size()>=3&&find(indexed_names.begin(),indexed_names.end(),w)==indexed_names.end())add(w,12);
        if(additional_blocks)for(auto&key:additional_blocks(r))add(key.first,key.second);
        if(i%2000000==0)cerr<<"Indexing "<<i<<"/"<<targets.size()<<"\n";
    }
    sort(index_entries.begin(),index_entries.end());cerr<<"Indexed "<<index_entries.size()<<" postings\n";
}
static vector<Hit> retrieve_enhanced(const Rec&r,size_t limit=160) {
    struct Block{size_t lo,hi;int type;};vector<Block> blocks;
    auto add=[&](const string&s,int type){if(s.size()<2)return;auto p=posting(hash_key(s,r.country,type));
        if(p.second>p.first&&p.second-p.first<=12000)blocks.push_back({p.first,p.second,type});};
    for(auto&w:words(r.name))if(w.size()>=3)add(w,1);for(auto&w:words(r.address))if(w.size()>=3)add(w,2);
    add(compact(lower(r.name)),3);auto pairs=[&](const string&s,int type){auto v=pair_words(s);
        for(size_t i=0;i<v.size();i++)for(size_t j=i+1;j<v.size();j++)add(pair_text(v[i],v[j]),type);};
    pairs(r.name,4);pairs(r.address,5);string cc=compact(join_words(core_words(r.name)));
    auto cw=core_words(r.name),aw=address_words(r.address);
    add(cc,6);add(join_words(aw),7);add(consonants(cc),8);
    for(auto&w:aw)if(w.size()>=3){add(w,2);add(w,9);}for(auto&w:number_blocks(r.address,aw))add(w,10);add(loose_core(cw),11);
    for(auto&w:cw)if(w.size()>=3){add(w,1);add(w,12);}
    if(additional_blocks)for(auto&key:additional_blocks(r))add(key.first,key.second);
    sort(blocks.begin(),blocks.end(),[](const Block&a,const Block&b){return a.hi-a.lo!=b.hi-b.lo?a.hi-a.lo<b.hi-b.lo:a.lo<b.lo;});
    // The same normalized/raw token can occur twice; never double count a posting.
    blocks.erase(unique(blocks.begin(),blocks.end(),[](const Block&a,const Block&b){return a.lo==b.lo&&a.hi==b.hi;}),blocks.end());
    unordered_map<uint32_t,float> scores;scores.reserve(3000);size_t budget=0;int counts[16]={};
    for(auto&b:blocks){size_t n=b.hi-b.lo;if(budget+n>30000||counts[b.type]>=8)continue;budget+=n;counts[b.type]++;
        float base=b.type==13?3.5f:b.type==14?1.f:b.type==15?1.8f:b.type==1||b.type==12?1.2f:b.type==2||b.type==9?1.0f:b.type==3||b.type==10?3.0f:b.type==4?2.0f:b.type==5?1.6f:b.type==6||b.type==7||b.type==11?4.0f:2.0f;
        float wt=base*(1+log1pf(800.0f/n));for(size_t j=b.lo;j<b.hi;j++)scores[index_entries[j].row]+=wt;
    }
    vector<Hit> hits;for(auto&x:scores)hits.push_back({x.first,x.second});
    auto better=[](const Hit&a,const Hit&b){return a.retrieval!=b.retrieval?a.retrieval>b.retrieval:a.row<b.row;};
    // A cheap similarity rerank reduces the influence of common-token postings.
    size_t rerank=max<size_t>(600,limit);if(hits.size()>rerank){nth_element(hits.begin(),hits.begin()+rerank,hits.end(),better);hits.resize(rerank);}
    auto un=core_words(r.name),ua=address_words(r.address);string uc=compact(join_words(un)),uac=join_words(ua);
    for(auto&h:hits){const auto&t=targets[h.row];auto vn=core_words(t.name),va=address_words(t.address);
        float ns=token_overlap(un,vn),as=token_overlap(ua,va),nd=dice3(uc,compact(join_words(vn)));
        h.retrieval+=12*nd+8*ns+8*as+12*nd*as;
    }
    if(hits.size()>limit){nth_element(hits.begin(),hits.begin()+limit,hits.end(),better);hits.resize(limit);}sort(hits.begin(),hits.end(),better);return hits;
}
struct BoostTree{vector<pair<int,float>> splits;vector<double> leaves;};
struct BoostModel {
    double threshold=0,bias=0,scale=1;vector<BoostTree> trees;
    explicit BoostModel(const string&path,int expected_features=BOOST_K){ifstream in(path);string version;int features,count;
        if(!(in>>version>>features>>threshold>>scale>>bias>>count)||version!="ERBOOST1"||features!=expected_features||count<=0)throw runtime_error("Invalid boosted model: "+path);
        for(int i=0;i<count;i++){BoostTree t;int depth;in>>depth;if(depth<1||depth>16)throw runtime_error("Invalid tree depth");
            for(int j=0;j<depth;j++){int feature;float border;in>>feature>>border;if(feature<0||feature>=expected_features)throw runtime_error("Invalid feature");t.splits.push_back({feature,border});}
            t.leaves.resize(1U<<depth);for(auto&x:t.leaves)in>>x;trees.push_back(std::move(t));}
        if(!in)throw runtime_error("Truncated boosted model");
    }
    template<class FeatureVector> double raw(const FeatureVector&x)const{double total=0;for(auto&t:trees){unsigned index=0;for(size_t k=0;k<t.splits.size();k++)index|=unsigned(x[t.splits[k].first]>t.splits[k].second)<<k;total+=t.leaves[index];}return scale*total+bias;}
};
static void export_features(const string&base,const string&outdir,int train_extra,int heldout,int limit) {
    filesystem::create_directories(outdir);
    // Preserve the original 12k/6k split; add training and untouched audit entities.
    auto original=sample_truth(base+"/dataset/train/train_ground_truth.tsv",18000);
    auto pool=sample_truth(base+"/dataset/train/train_ground_truth.tsv",max(60000,2*(train_extra+heldout)));
    unordered_set<uint32_t> used;for(auto&x:original)used.insert(x.first);
    vector<pair<uint32_t,vector<uint64_t>>> gt(original.begin(),original.begin()+12000);
    int added=0;for(auto&x:pool)if(!used.count(x.first)&&added<train_extra){gt.push_back(x);used.insert(x.first);added++;}
    int train_end=gt.size();gt.insert(gt.end(),original.begin()+12000,original.end());int tune_end=gt.size();
    added=0;for(auto&x:pool)if(!used.count(x.first)&&added<heldout){gt.push_back(x);used.insert(x.first);added++;}
    if(added!=heldout)throw runtime_error("Insufficient held-out queries");
    unordered_map<uint32_t,int> qmap;for(int i=0;i<(int)gt.size();i++)qmap[gt[i].first]=i;
    vector<Rec> queries(gt.size());ifstream in(base+"/dataset/train/train_source1.tsv");string line;getline(in,line);
    while(getline(in,line)){auto it=qmap.find(id_number(line));if(it!=qmap.end())parse_row(line,queries[it->second],1);}
    load_enhanced(base+"/dataset/train","train");
    ofstream data(outdir+"/features.f32",ios::binary),meta(outdir+"/pairs.u32",ios::binary),qfile(outdir+"/queries.tsv"),truthfile(outdir+"/truth.tsv"),schema(outdir+"/schema.json");
    if(!data||!meta||!qfile||!truthfile||!schema)throw runtime_error("Cannot create export files");
    schema<<"{\"feature_count\":"<<BOOST_K<<",\"candidate_limit\":"<<limit<<",\"train_end\":"<<train_end<<",\"tune_end\":"<<tune_end<<",\"query_count\":"<<gt.size()<<",\"features\":[";
    for(int i=0;i<20;i++){if(i)schema<<',';schema<<"\"legacy_"<<i<<"\"";}for(auto&s:EXTRA_NAMES)schema<<",\""<<s<<"\"";schema<<"]}\n";
    ifstream baseline_model(base+"/code/business_entity_resolution/model.txt");float baseline_threshold;int baseline_rules;array<float,K> baseline_weights;
    baseline_model>>baseline_threshold>>baseline_rules;for(auto&w:baseline_weights)baseline_model>>w;if(!baseline_model)throw runtime_error("Missing baseline model");
    qfile<<"query\tsource1_entity_id\tsplit\ttruth_count\tcountry\tbaseline_tp\tbaseline_fp\tbaseline_retrieved\tname\taddress\n";
    truthfile<<"query\tmatched_entity_id\tretrieved\n";uint64_t total=0,found=0,pairs=0;
    struct ExportRow{vector<BoostFeatures> x;vector<array<uint32_t,5>> meta;string query,truth;uint64_t found=0,total=0;};
    for(uint32_t start=0;start<gt.size();start+=128){
      vector<ExportRow> batch(min<size_t>(128,gt.size()-start));
      parallel_queries(batch.size(),[&](size_t offset){uint32_t qi=start+offset;auto&result=batch[offset];ostringstream qline,tline;
        const auto&a=queries[qi];if(a.name.empty()&&a.id==0)throw runtime_error("Missing query");Prepared u(a);
        unordered_set<uint64_t> truth(gt[qi].second.begin(),gt[qi].second.end()),seen;
        auto hits=retrieve_enhanced(a,limit);auto baseline=retrieve(a);unordered_set<uint32_t> old;for(auto&h:baseline)old.insert(h.row);
        int baseline_tp=0,baseline_fp=0,baseline_retrieved=0;
        for(auto&h:baseline){const auto&b=targets[h.row];bool y=truth.count((uint64_t(b.source)<<32)|b.id);auto x=features(a,b,h.retrieval);baseline_retrieved+=y;
            if(decide(x,predict_score(x,baseline_weights),baseline_threshold,baseline_rules)){if(y)baseline_tp++;else baseline_fp++;}}
        qline<<qi<<'\t'<<id_text(1,a.id)<<'\t'<<(qi<(uint32_t)train_end?"train":qi<(uint32_t)tune_end?"tune":"audit")<<'\t'<<truth.size()<<'\t'<<int(a.country)<<'\t'<<baseline_tp<<'\t'<<baseline_fp<<'\t'<<baseline_retrieved<<'\t'<<a.name<<'\t'<<a.address<<'\n';
        int rank=0;for(auto&h:hits){const auto&b=targets[h.row];uint64_t key=(uint64_t(b.source)<<32)|b.id;bool y=truth.count(key);Prepared v(b,false);
            result.x.push_back(extended_features(a,b,u,v,h.retrieval,rank++));
            result.meta.push_back({qi,b.id,b.source,uint32_t(y),uint32_t(old.count(h.row))});
            if(y){result.found++;seen.insert(key);}
        }
        for(auto key:truth)tline<<qi<<'\t'<<id_text(key>>32,uint32_t(key))<<'\t'<<seen.count(key)<<'\n';result.total=truth.size();
        result.query=qline.str();result.truth=tline.str();
      });
      for(auto&result:batch){data.write(reinterpret_cast<const char*>(result.x.data()),result.x.size()*sizeof(BoostFeatures));
        meta.write(reinterpret_cast<const char*>(result.meta.data()),result.meta.size()*sizeof(array<uint32_t,5>));
        qfile<<result.query;truthfile<<result.truth;found+=result.found;total+=result.total;pairs+=result.x.size();}
      if(start%1024==0)cerr<<"Export "<<start+batch.size()<<'/'<<gt.size()<<" pairs="<<pairs<<"\n";
    }
    if(!data||!meta)throw runtime_error("Feature export write failed");cerr<<"Exported "<<pairs<<" pairs\n";
}
static void predict_boost(const string&base,const string&path,const string&outdir,int limit,int max_queries) {
    BoostModel model(path);filesystem::create_directories(outdir);load_enhanced(base+"/dataset/test","test");
    ifstream in(base+"/dataset/test/test_source1.tsv");if(!in)throw runtime_error("Missing test Source 1");string line;getline(in,line);
    ofstream matches(outdir+"/matching_results.tsv"),candidates(outdir+"/candidate_pairs.tsv");if(!matches||!candidates)throw runtime_error("Cannot create outputs");
    matches<<"source1_entity_id\tmatched_entity_ids\n";candidates<<"source1_entity_id\tcandidate_entity_ids\n";uint64_t n=0,links=0;
    for(;;){vector<Rec> batch;while(batch.size()<128&&getline(in,line)){
        Rec a;if(!parse_row(line,a,1))throw runtime_error("Malformed query");batch.push_back(std::move(a));
        if(max_queries>0&&n+batch.size()>=(uint64_t)max_queries)break;}
      if(batch.empty())break;struct Prediction{string matches,candidates;uint64_t links=0;};vector<Prediction> results(batch.size());
      parallel_queries(batch.size(),[&](size_t i){const auto&a=batch[i];auto&result=results[i];Prepared u(a);auto hits=retrieve_enhanced(a,limit);string m,c;int rank=0;
        for(auto&h:hits){const auto&b=targets[h.row];Prepared v(b,false);auto x=extended_features(a,b,u,v,h.retrieval,rank++);string id=id_text(b.source,b.id);
            if(!c.empty())c+=',';c+=id;if(model.raw(x)>=model.threshold){if(!m.empty())m+=',';m+=id;result.links++;}}
        result.matches=std::move(m);result.candidates=std::move(c);
      });
      for(size_t i=0;i<batch.size();i++){string id=id_text(1,batch[i].id);matches<<id<<'\t'<<results[i].matches<<'\n';candidates<<id<<'\t'<<results[i].candidates<<'\n';links+=results[i].links;}
      n+=batch.size();if(n%10240==0)cerr<<"Predicted "<<n<<" queries; "<<links<<" links\n";if(max_queries>0&&n>=(uint64_t)max_queries)break;
    }
    if(!matches||!candidates)throw runtime_error("Output write failed");cerr<<"Finished "<<n<<" queries; "<<links<<" links\n";
}
static void verify_scores(const string&modelpath,const string&featurespath,const string&outpath) {
    BoostModel model(modelpath);ifstream in(featurespath,ios::binary);ofstream out(outpath,ios::binary);if(!in||!out)throw runtime_error("Cannot open score files");BoostFeatures x;
    while(in.read(reinterpret_cast<char*>(x.data()),sizeof(x))){double p=model.raw(x);out.write(reinterpret_cast<const char*>(&p),sizeof(p));}
    if(!in.eof()||in.gcount()!=0)throw runtime_error("Partial feature row");
}
#ifndef BOOST_NO_MAIN
int main(int argc,char**argv) {try{
    if(argc>=4&&string(argv[1])=="export")export_features(argv[2],argv[3],argc>4?stoi(argv[4]):18000,argc>5?stoi(argv[5]):6000,argc>6?stoi(argv[6]):160);
    else if(argc>=5&&string(argv[1])=="predict")predict_boost(argv[2],argv[3],argv[4],argc>5?stoi(argv[5]):160,argc>6?stoi(argv[6]):0);
    else if(argc==5&&string(argv[1])=="score")verify_scores(argv[2],argv[3],argv[4]);
    else {cerr<<"Usage: resolver_boost export BASE OUTPUT [EXTRA_TRAIN=18000 AUDIT=6000 CANDIDATES=160]\n"
        <<"       resolver_boost predict BASE MODEL OUTPUT [CANDIDATES=160 MAX_QUERIES=0]\n"
        <<"       resolver_boost score MODEL FEATURES_F32 SCORES_F64\n";return 2;}
    return 0;
}catch(const exception&e){cerr<<"Error: "<<e.what()<<'\n';return 1;}}
#endif
