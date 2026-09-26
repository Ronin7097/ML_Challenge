static constexpr int FULL_K=BOOST_K+ADVANCED_K, CONTEXT_K=23;
using FullFeatures=array<float,FULL_K>;

static vector<bool> choose_matches(const vector<double>&scores,double threshold,bool expected){
    vector<bool> chosen(scores.size(),false);if(!expected){for(size_t i=0;i<scores.size();i++)chosen[i]=scores[i]>=threshold;return chosen;}
    vector<size_t> order(scores.size());iota(order.begin(),order.end(),0);stable_sort(order.begin(),order.end(),[&](size_t a,size_t b){return scores[a]>scores[b];});
    vector<double> p;double total=0,log_empty=0;for(auto i:order){double value=1/(1+exp(-max(-50.,min(50.,scores[i]))));p.push_back(value);total+=value;log_empty+=log1p(-min(value,1-1e-15));}
    double best=exp(log_empty),sum=0;size_t count=0;
    for(size_t i=0;i<p.size();i++){sum+=p[i];double utility=1.25*sum/(i+1+.25*total);if(utility>best){best=utility;count=i+1;}}
    for(size_t i=0;i<count;i++)chosen[order[i]]=true;return chosen;
}

static vector<vector<float>> make_context(const vector<FullFeatures>&features,const vector<double>&scores,const vector<uint8_t>&sources,
                                         const vector<ReferenceFeatures>&reference={}){
    size_t n=features.size();vector<vector<float>> result(n);vector<double> p(n);vector<size_t> order(n),rank(n);iota(order.begin(),order.end(),0);
    double total=0;for(size_t i=0;i<n;i++){p[i]=1/(1+exp(-max(-40.,min(40.,scores[i]))));total+=p[i];}
    stable_sort(order.begin(),order.end(),[&](size_t a,size_t b){return p[a]>p[b];});for(size_t i=0;i<n;i++)rank[order[i]]=i;
    array<double,5> cutoffs={.25,.5,.75,.9,.99};array<int,5> counts{};double source_sum[4]={};int source_50[4]={},source_90[4]={};
    for(size_t i=0;i<n;i++){for(size_t k=0;k<cutoffs.size();k++)counts[k]+=p[i]>=cutoffs[k];source_sum[sources[i]]+=p[i];source_50[sources[i]]+=p[i]>=.5;source_90[sources[i]]+=p[i]>=.9;}
    for(size_t i=0;i<n;i++){
        auto&out=result[i];out.assign(features[i].begin(),features[i].end());auto put=[&](double x){out.push_back(float(x));};
        array<double,4> best{};size_t at=0;for(auto j:order)if(j!=i){best[at++]=p[j];if(at==best.size())break;}
        put(scores[i]);put(p[i]);put(log1p(double(rank[i])));for(auto value:best)put(value);put(total-p[i]);
        for(size_t k=0;k<cutoffs.size();k++)put(counts[k]-(p[i]>=cutoffs[k]));
        auto same=sources[i],other=uint8_t(same==2?3:2);double same_best=0,other_best=0;
        for(auto j:order)if(j!=i&&sources[j]==same){same_best=p[j];break;}for(auto j:order)if(sources[j]==other){other_best=p[j];break;}
        put(same_best);put(source_sum[same]-p[i]);put(source_50[same]-(p[i]>=.5));put(source_90[same]-(p[i]>=.9));
        put(other_best);put(source_sum[other]);put(source_50[other]);put(source_90[other]);put(p[i]/max(total,1e-30));put(p[i]-best[0]);
        if(out.size()!=FULL_K+CONTEXT_K)throw runtime_error("Context feature schema mismatch");
        if(!reference.empty())out.insert(out.end(),reference[i].begin(),reference[i].end());
    }
    return result;
}
