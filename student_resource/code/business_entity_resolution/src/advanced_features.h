#include "transliteration_data.h"

static string romanize(const string& input) {
    string s=fold_text(input),out;bool inherent=false;
    for(size_t i=0;i<s.size();) {
        size_t start=i;uint32_t c=uint8_t(s[i++]);
        if(c>=192&&c<224&&i<s.size())c=((c&31)<<6)|(uint8_t(s[i++])&63);
        else if(c>=224&&c<240&&i+1<s.size()){c=((c&15)<<12)|((uint8_t(s[i])&63)<<6)|(uint8_t(s[i+1])&63);i+=2;}
        auto it=ROMAN_RULES.find(c);
        if(it!=ROMAN_RULES.end()) {
            auto&rule=it->second;
            if((rule.second==3||rule.second==4)&&inherent&&!out.empty())out.pop_back();
            out+=rule.first;inherent=rule.second==2;
        } else if(c==0x200c||c==0x200d||(c>=0x900&&c<0xD80)||(c>=0x300&&c<=0x36f)) {
            // Ignore zero-width joiners, accent marks, and unsupported script marks.
        } else {out+=s.substr(start,i-start);inherent=false;}
    }
    return out;
}
static string phonetic(const string&s) {
    string z;for(size_t i=0;i<s.size();i++){
        char c=s[i];if(c<'a'||c>'z')continue;
        if(c=='p'&&i+1<s.size()&&s[i+1]=='h'){c='f';i++;}
        if(c=='x'){z+="ks";continue;}
        if(c=='c'||c=='q')c='k';else if(c=='w')c='v';else if(c=='z')c='s';
        if(string("aeiouh").find(c)!=string::npos)continue;
        if(z.empty()||z.back()!=c)z+=c;
    }return z;
}
static vector<string> sound_words(const string&name) {
    static const unordered_set<string> legal={"lmt d","lmtd","prvt","prv t","prv","pvt","ltd","llk","llp","lk","kp","krp","krprtn","nkrprtd","kmpn","th","nd","plk"};
    vector<string> out;for(auto&w:core_words(romanize(name))){string p=phonetic(w);if(!p.empty()&&!legal.count(p))out.push_back(p);}
    sort(out.begin(),out.end());out.erase(unique(out.begin(),out.end()),out.end());return out;
}
static void enable_sound_blocks(){additional_blocks=[](const Rec&r){
    auto v=sound_words(r.name);vector<pair<string,uint8_t>> out;
    if(!v.empty())out.push_back({join_words(v),13});
    for(auto&w:v)if(w.size()>=3)out.push_back({w,14});
    if(v.size()<=8)for(size_t i=0;i<v.size();i++)for(size_t j=i+1;j<v.size();j++)if(v[i].size()+v[j].size()>=5)out.push_back({pair_text(v[i],v[j]),15});
    return out;
};}
static vector<string> digit_runs(const string&s) {
    vector<string> out;string current;
    auto finish=[&]{if(!current.empty()){size_t first=current.find_first_not_of('0');out.push_back(first==string::npos?"0":current.substr(first));current.clear();}};
    for(char c:s){if(c>='0'&&c<='9')current+=c;else finish();}finish();
    sort(out.begin(),out.end());out.erase(unique(out.begin(),out.end()),out.end());return out;
}
static vector<string> mixed_numbers(const string&s) {
    vector<string> out;for(auto&w:words(s))if(any_of(w.begin(),w.end(),[](char c){return c>='0'&&c<='9';}))out.push_back(w);return out;
}
struct AdvancedPrepared {
    vector<string> roman, sound, digits, mixed, addr;string rc,sc,ac;
    explicit AdvancedPrepared(const Rec&r){roman=core_words(romanize(r.name));sound=sound_words(r.name);digits=digit_runs(r.address);mixed=mixed_numbers(r.address);
        addr=address_words(romanize(r.address));rc=compact(join_words(roman));sc=join_words(sound);ac=join_words(addr);}
};
static const vector<string> ADVANCED_NAMES={
    "roman_exact","roman_dice","roman_edit","roman_overlap","roman_jaccard","roman_fuzzy_query","roman_fuzzy_target",
    "sound_exact","sound_dice","sound_edit","sound_overlap","sound_jaccard","sound_fuzzy_query","sound_fuzzy_target",
    "sound_missing_query","sound_missing_target","digits_overlap","digits_jaccard","digits_query_only","digits_target_only",
    "digits_fuzzy_query","digits_fuzzy_target","mixed_overlap","mixed_jaccard","mixed_fuzzy_query","mixed_fuzzy_target",
    "roman_address_dice","roman_address_overlap","roman_address_jaccard","sound_query_tokens","sound_target_tokens"
};
static constexpr int ADVANCED_K=31;
using AdvancedFeatures=array<float,ADVANCED_K>;
static AdvancedFeatures advanced_features(const AdvancedPrepared&a,const AdvancedPrepared&b){
    AdvancedFeatures out{};int k=0;auto put=[&](float x){out.at(k++)=x;};
    put(!a.rc.empty()&&a.rc==b.rc);put(dice3(a.rc,b.rc));put(edit_similarity(a.rc,b.rc));put(token_overlap(a.roman,b.roman));put(token_jaccard(a.roman,b.roman));
    auto f=fuzzy_overlap(a.roman,b.roman),g=fuzzy_overlap(b.roman,a.roman);put(f.first);put(g.first);
    put(!a.sc.empty()&&a.sc==b.sc);put(dice2(a.sc,b.sc));put(edit_similarity(a.sc,b.sc));put(token_overlap(a.sound,b.sound));put(token_jaccard(a.sound,b.sound));
    f=fuzzy_overlap(a.sound,b.sound);g=fuzzy_overlap(b.sound,a.sound);put(f.first);put(g.first);put(f.second);put(g.second);
    put(token_overlap(a.digits,b.digits));put(token_jaccard(a.digits,b.digits));
    int shared=0;for(auto&w:a.digits)shared+=binary_search(b.digits.begin(),b.digits.end(),w);put(a.digits.size()-shared);put(b.digits.size()-shared);
    f=fuzzy_overlap(a.digits,b.digits);g=fuzzy_overlap(b.digits,a.digits);put(f.first);put(g.first);
    put(token_overlap(a.mixed,b.mixed));put(token_jaccard(a.mixed,b.mixed));f=fuzzy_overlap(a.mixed,b.mixed);g=fuzzy_overlap(b.mixed,a.mixed);put(f.first);put(g.first);
    put(dice3(a.ac,b.ac));put(token_overlap(a.addr,b.addr));put(token_jaccard(a.addr,b.addr));put(a.sound.size());put(b.sound.size());
    if(k!=ADVANCED_K)throw runtime_error("Advanced feature schema mismatch");return out;
}
