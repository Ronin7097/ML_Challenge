#define ADVANCED_NO_MAIN
#include "advanced.cpp"
#include <cassert>

int main(){
    assert(sound_words("राम मार्केटिंग प्राइवेट लिमिटेड")==sound_words("Ram Marketing Private Limited"));
    assert(sound_words("ಸೆವೆನ್ ಎಕ್ಸ್‌ಪೋರ್ಟ್ಸ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್")==sound_words("Seven Exports Private Limited"));
    assert(core_words("École Française SARL")==core_words("Ecole Francaise"));
    assert(digit_runs("Flat 27K/18K, #00123")==vector<string>({"123","18","27"}));
    assert(ReferenceIndex::subsequence("234","1234"));
    assert(!ReferenceIndex::subsequence("1235","1234"));
    auto dropped=ReferenceIndex::numeric_evidence({"1234"},{"234"});
    auto substituted=ReferenceIndex::numeric_evidence({"1234"},{"1235"});
    assert(dropped[0]==1&&substituted[0]==0);
    cout<<"PASS: cross-script names, accents, alphanumeric addresses, digit omission versus substitution\n";
}
