export module RNN_LSTM;

import Matrix;
import  RevAutoDiffEngine;

import <vector>;
import <cmath>;
import <iostream>;
import <cassert>;
import <memory>;
using namespace std;

shared_ptr<Node> detach(const shared_ptr<Node>& n) {
	return make_shared<Node>(n->DATA);
}

struct QUADRAAPLET {
	shared_ptr<Node> F, I, C, O;
	QUADRAAPLET(shared_ptr<Node> f, shared_ptr<Node> i, shared_ptr<Node> c, shared_ptr<Node> o)
		: F(f), I(i), C(c), O(o) {
	}
	QUADRAAPLET() = default;
};

class LSTM_Node {
public:
	pair<shared_ptr<Node>, shared_ptr<Node>> forward(QUADRAAPLET& Ws, QUADRAAPLET& Us, QUADRAAPLET& Bs, shared_ptr<Node>& xt, shared_ptr<Node>& ht_1, shared_ptr<Node>& ct_1) {
		//Forget Gate - Controls how much of the past cell state to keep.
		shared_ptr<Node> Ft = sigmoid(xt * Ws.F + ht_1 * Us.F + Bs.F);
		//fₜ ∈[0, 1]: Closer to 1 means keep information, closer to 0 means forget.

		//Input Gate - Controls what new information to add to the cell state.
		shared_ptr<Node> It = sigmoid(xt * Ws.I + ht_1 * Us.I + Bs.I);
		shared_ptr<Node> Ct_cap = tanh(xt * Ws.C + ht_1 * Us.C + Bs.C);
		shared_ptr<Node> Ct = elementaryProduct(Ft, ct_1) + elementaryProduct(It, Ct_cap);
		//Combines retained memory and new information

		//Output Gate - What to send as output and Decides what part of the cell state becomes the hidden state.
		shared_ptr<Node> Ot = sigmoid(xt * Ws.O + ht_1 * Us.O + Bs.O);
		shared_ptr<Node> Ht = elementaryProduct(Ot, tanh(Ct));
		//hₜ is the final output passed to next timestep

		return { Ht, Ct };
	}
};

export class LSTM_RNN {
	int CELL_SIZE, HIDDEN_SIZE, VOCAB_SIZE;
	unordered_map<int, int> MELODY_VOCAB;

	shared_ptr<LSTM_Node> RNN;
	QUADRAAPLET Ws, Us, Bs;
	shared_ptr<Node> Wo, Bo;

public:
	LSTM_RNN(int cell, int hidden, unordered_map<int, int>& vocab)
		: CELL_SIZE(cell), HIDDEN_SIZE(hidden), MELODY_VOCAB(vocab), VOCAB_SIZE(vocab.size()) {

		auto randM = [&](int r, int c) { return make_shared<Node>(Matrix(r, c, "random")); };
		auto zeroM = [&](int r, int c) { return make_shared<Node>(Matrix(r, c, 0)); };

		Ws = QUADRAAPLET(randM(cell, hidden), randM(cell, hidden), randM(cell, hidden), randM(cell, hidden));
		Us = QUADRAAPLET(randM(hidden, hidden), randM(hidden, hidden), randM(hidden, hidden), randM(hidden, hidden));
		Bs = QUADRAAPLET(zeroM(1, hidden), zeroM(1, hidden), zeroM(1, hidden), zeroM(1, hidden));

		Wo = randM(hidden, VOCAB_SIZE);
		Bo = zeroM(1, VOCAB_SIZE);
		RNN = make_shared<LSTM_Node>();
	}

	void Train(vector<vector<int>>& dataset, vector<int>& targets,
		int epochs, double lr, double clipThresh, double reg, int truncateStep) {

		for (int epoch = 0; epoch < epochs; ++epoch) {
			double avgLoss = 0;
			for (int i = 0; i < dataset.size(); ++i) {
				showProgressBar(i + 1, dataset.size());
				shared_ptr<Node> ct = make_shared<Node>(Matrix(1, HIDDEN_SIZE, 0));
				shared_ptr<Node> ht = make_shared<Node>(Matrix(1, HIDDEN_SIZE, 0));

				for (int t = 0; t < dataset[i].size(); ++t) {
					shared_ptr<Node> xt = make_shared<Node>(oneHotMatrix(MELODY_VOCAB, dataset[i][t]));
					tie(ht, ct) = RNN->forward(Ws, Us, Bs, xt, ht, ct);
					if (truncateStep > 0 && t % truncateStep == 0) {
						ht = detach(ht);
						ct = detach(ct);
					}
				}

				shared_ptr<Node> out = ht * Wo + Bo;
				auto [prob, loss] = Softmaxed_CCE(out, oneHotMatrix(MELODY_VOCAB, targets[i]));
				avgLoss += loss;

				clipGrads(Wo, clipThresh);
				clipGrads(Bo, clipThresh);
				clipAll(clipThresh);

				// Weight update
				Wo->DATA = Wo->DATA - lr * (Wo->GRADIENT + reg * Wo->DATA);
				Bo->DATA = Bo->DATA - lr * Bo->GRADIENT;
				updateWeights(lr, reg);
				resetGrads();
			}
			cout << "\nLoss: " << avgLoss / dataset.size() << " | Epoch: " << epoch + 1 << endl;
		}
	}

	Matrix predict(const vector<int>& data, double temperature = 0.8) {
		shared_ptr<Node> ct_1 = make_shared<Node>(Matrix(1, HIDDEN_SIZE, 0));
		shared_ptr<Node> ht_1 = make_shared<Node>(Matrix(1, HIDDEN_SIZE, 0));

		for (int k = 0; k < data.size(); ++k) {
			shared_ptr<Node> x = make_shared<Node>(oneHotMatrix(MELODY_VOCAB, data[k]));
			auto [Ht, Ct] = RNN->forward(Ws, Us, Bs, x, ht_1, ct_1);
			ht_1 = Ht;
			ct_1 = Ct;
		}

		shared_ptr<Node> Output = ht_1 * Wo + Bo;
		Matrix Probabilities = softmax(Output->DATA);

		int idx = sampleFromDistribution(Probabilities, temperature);
		Matrix oneHot(1, Probabilities.COLUMN_SIZE, 0);
		oneHot.set(0, idx, 1);
		return oneHot;
	}


private:
	int sampleFromDistribution(const Matrix& softmaxOutput, double temperature) {
		std::vector<double> probs;
		double sum = 0.0;

		for (int i = 0; i < softmaxOutput.COLUMN_SIZE; ++i) {
			double val = std::pow(softmaxOutput.get(0, i), 1.0 / temperature);
			probs.push_back(val);
			sum += val;
		}

		for (auto& p : probs) p /= sum;

		double r = random_uniform(0.0, 1.0);
		double cumulative = 0.0;

		for (int i = 0; i < probs.size(); ++i) {
			cumulative += probs[i];
			if (r <= cumulative)
				return i;
		}
		return static_cast<int>(probs.size()) - 1;
	}

	void clipGrads(shared_ptr<Node>& m, double thresh) {
		for (int i = 0; i < m->GRADIENT.ROW_SIZE; ++i)
			for (int j = 0; j < m->GRADIENT.COLUMN_SIZE; ++j) {
				double v = m->GRADIENT.get(i, j);
				m->GRADIENT.set(i, j, max(-thresh, min(thresh, v)));
			}
	}

	void clipAll(double t) {
		clipGrads(Ws.F, t); clipGrads(Us.F, t); clipGrads(Bs.F, t);
		clipGrads(Ws.I, t); clipGrads(Us.I, t); clipGrads(Bs.I, t);
		clipGrads(Ws.C, t); clipGrads(Us.C, t); clipGrads(Bs.C, t);
		clipGrads(Ws.O, t); clipGrads(Us.O, t); clipGrads(Bs.O, t);
	}

	void resetGrads() {
		Ws.F->GRADIENT = Matrix(CELL_SIZE, HIDDEN_SIZE, 0);
		Ws.I->GRADIENT = Matrix(CELL_SIZE, HIDDEN_SIZE, 0);
		Ws.C->GRADIENT = Matrix(CELL_SIZE, HIDDEN_SIZE, 0);
		Ws.O->GRADIENT = Matrix(CELL_SIZE, HIDDEN_SIZE, 0);

		Us.F->GRADIENT = Matrix(HIDDEN_SIZE, HIDDEN_SIZE, 0);
		Us.I->GRADIENT = Matrix(HIDDEN_SIZE, HIDDEN_SIZE, 0);
		Us.C->GRADIENT = Matrix(HIDDEN_SIZE, HIDDEN_SIZE, 0);
		Us.O->GRADIENT = Matrix(HIDDEN_SIZE, HIDDEN_SIZE, 0);

		Bs.F->GRADIENT = Matrix(1, HIDDEN_SIZE, 0);
		Bs.I->GRADIENT = Matrix(1, HIDDEN_SIZE, 0);
		Bs.C->GRADIENT = Matrix(1, HIDDEN_SIZE, 0);
		Bs.O->GRADIENT = Matrix(1, HIDDEN_SIZE, 0);

		Wo->GRADIENT = Matrix(HIDDEN_SIZE, VOCAB_SIZE, 0);
		Bo->GRADIENT = Matrix(1, VOCAB_SIZE, 0);
	}

	void updateWeights(double lr, double lambda) {
		auto update = [&](shared_ptr<Node>& w) {
			w->DATA = w->DATA - lr * (w->GRADIENT + lambda * w->DATA);
			};
		update(Ws.F); update(Us.F); update(Bs.F);
		update(Ws.I); update(Us.I); update(Bs.I);
		update(Ws.C); update(Us.C); update(Bs.C);
		update(Ws.O); update(Us.O); update(Bs.O);
	}
};