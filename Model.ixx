import Matrix;
import RNN_LSTM;

#include <vector>
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <unordered_map>
#include <set>
using namespace std;

unordered_map<int, int> melody2idx;
vector<int> idx2melody;
int vocabSize = 0;

static pair<vector<vector<int>>, vector<int>> load_note_sequences(const string& filename) {
    vector<vector<int>> sequences;
    vector<int> outputs;

    ifstream infile(filename);
    string line;

    while (getline(infile, line)) {
        istringstream iss(line);
        vector<int> seq;
        int note;

        while (iss >> note) {
            seq.push_back(note);
        }

        if (seq.size() > 1) {
            int last_note = seq.back();
            seq.pop_back();
            sequences.push_back(seq);
            outputs.push_back(last_note);
        }
    }

    return { sequences, outputs };
}

static void buildVocabulary(const vector<vector<int>>& tokens) {
    set<int> unique_values;
    for (const auto& row : tokens)
        unique_values.insert(row.begin(), row.end());

    int idx = 0;
    for (int melody : unique_values) {
        melody2idx[melody] = idx++;
        idx2melody.push_back(melody);
    }
    vocabSize = static_cast<int>(melody2idx.size());
}

static int getMelody(const Matrix& oneHotOutput) {
    for (int i = 0; i < oneHotOutput.COLUMN_SIZE; i++) {
        if (oneHotOutput.get(0, i) == 1)
            return idx2melody[i];
    }
    return -1; // fallback
}

int main() {
    auto [dataset, oneHotOutput] = load_note_sequences("Melody.txt");
    buildVocabulary(dataset);

    double learningRate = 0.05;
    double threshold = 1.0;
    double L2_Strength = 0.05;
    int truncateStep = 3;
    int epochs = 6;

    LSTM_RNN model(vocabSize, vocabSize, melody2idx);
    model.Train(dataset, oneHotOutput, epochs, learningRate, threshold, L2_Strength, truncateStep);

    // Generate sequence from last known input
    vector<int> inputSeq = dataset.back();
    cout << "Generated Sequence: ";
    for (int i = 0; i < 500; i++) {
        Matrix out = model.predict(inputSeq);
        int nextMelody = getMelody(out);
        cout << nextMelody << " ";

        inputSeq.erase(inputSeq.begin());
        inputSeq.push_back(nextMelody);
    }
    cout << endl;

    return 0;
};