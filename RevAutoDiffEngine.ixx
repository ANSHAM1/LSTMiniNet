export module RevAutoDiffEngine;

import Matrix;

import <vector>;
import <memory>;
import <unordered_set>;
import <functional>;
#include <iostream>
using namespace std;

export class Node {
public:
	Matrix DATA;
	Matrix GRADIENT;

	vector<shared_ptr<Node>> PARENTS;
	function<void()> backward;

	Node(const Matrix& data) :
		DATA(data),
		GRADIENT(Matrix(data.ROW_SIZE, data.COLUMN_SIZE, 0)),
		backward([]() {}) {
	}
};

export shared_ptr<Node> operator*(const shared_ptr<Node>& A, const shared_ptr<Node>& B) {
	Matrix M = A->DATA * B->DATA;
	auto result = make_shared<Node>(M);
	result->PARENTS = { A, B };

	result->backward = [A, B, result]() {
		A->GRADIENT = A->GRADIENT + (result->GRADIENT * B->DATA.Transpose());
		B->GRADIENT = B->GRADIENT + (A->DATA.Transpose() * result->GRADIENT);
		A->backward();
		B->backward();
		};

	return result;
}

export shared_ptr<Node> operator+(const shared_ptr<Node>& A, const shared_ptr<Node>& B) {
	Matrix M = A->DATA + B->DATA;
	auto result = make_shared<Node>(M);
	result->PARENTS = { A, B };

	result->backward = [A, B, result]() {
		A->GRADIENT = A->GRADIENT + result->GRADIENT;
		B->GRADIENT = B->GRADIENT + result->GRADIENT;
		A->backward();
		B->backward();
		};

	return result;
}

export shared_ptr<Node> operator-(const shared_ptr<Node>& A, const shared_ptr<Node>& B) {
	Matrix M = A->DATA - B->DATA;
	auto result = make_shared<Node>(M);
	result->PARENTS = { A, B };

	result->backward = [A, B, result]() {
		A->GRADIENT = A->GRADIENT + result->GRADIENT;
		B->GRADIENT = B->GRADIENT - result->GRADIENT;
		A->backward();
		B->backward();
		};

	return result;
}

export shared_ptr<Node> elementaryProduct(const shared_ptr<Node>& A, const shared_ptr<Node>& B) {
	Matrix M = std_Matrix::elementaryProduct(A->DATA, B->DATA);
	auto result = make_shared<Node>(M);
	result->PARENTS = { A, B };

	result->backward = [A, B, result]() {
		A->GRADIENT = A->GRADIENT + std_Matrix::elementaryProduct(result->GRADIENT, B->DATA);
		B->GRADIENT = B->GRADIENT + std_Matrix::elementaryProduct(result->GRADIENT, A->DATA);
		A->backward();
		B->backward();
		};

	return result;
}

export shared_ptr<Node> relu(const shared_ptr<Node>& A) {
	Matrix M = A->DATA.reluMat();
	auto result = make_shared<Node>(M);
	result->PARENTS = { A };

	result->backward = [A, result]() {
		A->GRADIENT = A->GRADIENT + std_Matrix::elementaryProduct(result->GRADIENT, result->DATA.dReluMat());
		A->backward();
		};

	return result;
}

export shared_ptr<Node> tanh(const shared_ptr<Node>& A) {
	Matrix M = A->DATA.tanhMat();
	auto result = make_shared<Node>(M);
	result->PARENTS = { A };

	result->backward = [A, result]() {
		A->GRADIENT = A->GRADIENT + std_Matrix::elementaryProduct(result->GRADIENT, result->DATA.dTanhMat());
		A->backward();
		};

	return result;
}

export shared_ptr<Node> sigmoid(const shared_ptr<Node>& A) {
	Matrix M = A->DATA.sigmoidMat();
	auto result = make_shared<Node>(M);
	result->PARENTS = { A };

	result->backward = [A, result]() {
		A->GRADIENT = A->GRADIENT + std_Matrix::elementaryProduct(result->GRADIENT, result->DATA.dSigmoidMat());
		A->backward();
		};

	return result;
}

export pair<Matrix, double> Softmaxed_CCE(const shared_ptr<Node>& A, const Matrix& target) {
	Matrix softmaxed = softmax(A->DATA);
	double loss = std_Loss::CCE(softmaxed, target);

	A->GRADIENT = std_Loss::dSoft_CCE(softmaxed, target);
	A->backward();

	return { softmaxed, loss };
}

export double CCE(const shared_ptr<Node>& A, const Matrix& target) {
	double loss = std_Loss::CCE(A->DATA, target);
	A->GRADIENT = std_Loss::dCCE(A->DATA, target);
	A->backward();
	return loss;
}

export double BCE(const shared_ptr<Node>& A, const Matrix& target) {
	double loss = std_Loss::BCE(A->DATA, target);
	A->GRADIENT = std_Loss::dBCE(A->DATA, target);
	A->backward();
	return loss;
}

export double MSE(const shared_ptr<Node>& A, const Matrix& target) {
	double loss = std_Loss::MSE(A->DATA, target);
	A->GRADIENT = std_Loss::dMSE(A->DATA, target);
	A->backward();
	return loss;
}
