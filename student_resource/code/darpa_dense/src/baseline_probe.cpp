// Evaluate selected queries with the frozen DARPA model against the full pool.
// The original predictor is included without changing its code or model files.
#define main darpa_original_advanced_main
#include "advanced.cpp"
#undef main

int main(int argc, char** argv) {
    try {
        if (argc != 7) throw runtime_error(
            "Usage: baseline_probe DATASET_DIR train|test MODELS IDS_FILE OUTPUT THREADS");
        const string data = argv[1], split = argv[2], models = argv[3], out = argv[5];
        if (split != "train" && split != "test") throw runtime_error("Invalid split");
        setenv("ER_THREADS", argv[6], 1);
        unordered_set<string> wanted;
        ifstream ids(argv[4]); string line;
        if (!ids) throw runtime_error("Cannot open requested IDs");
        while (getline(ids, line)) if (!line.empty()) wanted.insert(line);
        if (wanted.empty()) throw runtime_error("No requested IDs");
        ifstream header(models + "/context.boost"); string version; int count = 0;
        header >> version >> count;
        if (!header || version != "ERBOOST1" || count != FULL_K + CONTEXT_K + REFERENCE_K)
            throw runtime_error("Unexpected frozen context model schema");
        BoostModel pair(models + "/pair.boost", FULL_K), context_model(models + "/context.boost", count);
        const string root = data + "/" + split;
        ReferenceIndex references(root + "/" + split + "_source1.tsv");
        load_enhanced(root, split);  // Frozen config has phonetic_blocking=false.
        filesystem::create_directories(out);
        if (filesystem::exists(out + "/complete.json")) throw runtime_error("Output already complete");
        ofstream matching(out + "/matching_results.tsv"), candidates(out + "/candidate_pairs.tsv");
        if (!matching || !candidates) throw runtime_error("Cannot open outputs");
        matching << "source1_entity_id\tmatched_entity_ids\n";
        candidates << "source1_entity_id\tcandidate_entity_ids\n";
        ifstream input(root + "/" + split + "_source1.tsv"); getline(input, line);
        vector<Rec> queries;
        while (getline(input, line)) {
            if (!wanted.count(line.substr(0, line.find('\t')))) continue;
            Rec r; if (!parse_row(line, r, 1)) throw runtime_error("Malformed query");
            queries.push_back(std::move(r));
        }
        if (queries.size() != wanted.size()) throw runtime_error("Requested IDs absent or duplicated");
        for (size_t start = 0; start < queries.size(); start += 128) {
            const size_t n = min(size_t(128), queries.size() - start);
            struct Result { string matches, candidates; }; vector<Result> results(n);
            parallel_queries(n, [&](size_t i) {
                auto& q = queries[start+i]; Prepared u(q); AdvancedPrepared au(q);
                auto hits = retrieve_enhanced(q, 160);
                vector<FullFeatures> x(hits.size()); vector<double> scores(hits.size());
                vector<uint8_t> sources; vector<ReferenceFeatures> reverse(hits.size());
                for (size_t j = 0; j < hits.size(); ++j) {
                    auto& t = targets[hits[j].row]; Prepared v(t, false); AdvancedPrepared av(t);
                    auto f = extended_features(q, t, u, v, hits[j].retrieval, j);
                    auto extra = advanced_features(au, av);
                    copy(f.begin(), f.end(), x[j].begin());
                    copy(extra.begin(), extra.end(), x[j].begin()+BOOST_K);
                    sources.push_back(t.source); scores[j] = pair.raw(x[j]);
                    reverse[j] = references.features(q, t, x[j], scores[j], pair);
                }
                auto context = make_context(x, scores, sources, reverse); vector<double> final_scores;
                for (auto& f : context) final_scores.push_back(context_model.raw(f));
                auto decisions = choose_matches(final_scores, context_model.threshold, true);
                for (size_t j = 0; j < hits.size(); ++j) {
                    auto& t = targets[hits[j].row]; auto id = id_text(t.source, t.id);
                    if (!results[i].candidates.empty()) results[i].candidates += ',';
                    results[i].candidates += id;
                    if (decisions[j]) {
                        if (!results[i].matches.empty()) results[i].matches += ',';
                        results[i].matches += id;
                    }
                }
            });
            for (size_t i = 0; i < n; ++i) {
                auto id = id_text(1, queries[start+i].id);
                matching << id << '\t' << results[i].matches << '\n';
                candidates << id << '\t' << results[i].candidates << '\n';
            }
            cerr << "Baseline probe " << start+n << '/' << queries.size() << '\n';
        }
        matching.close(); candidates.close();
        if (!matching || !candidates) throw runtime_error("Output write failed");
        ofstream done(out + "/complete.json");
        done << "{\"queries\":" << queries.size() << ",\"full_target_pool\":true,\"full_owner_pool\":true}\n";
        if (!done) throw runtime_error("Completion write failed");
    } catch (const exception& e) { cerr << e.what() << '\n'; return 1; }
}
