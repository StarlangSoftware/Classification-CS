using System;
using System.IO;
using Classification.Parameter;
using Classification.Performance;
using Math;

namespace Classification.Model
{
    public class MultiLayerPerceptronModel : LinearPerceptronModel
    {
        private Matrix _V;
        private ActivationFunction _activationFunction;

        /**
         * <summary> The allocateWeights method allocates layers' weights of Matrix W and V.</summary>
         *
         * <param name="H">Integer value for weights.</param>
         */
        private void AllocateWeights(int H, Random random)
        {
            W = AllocateLayerWeights(H, d + 1, random);
            _V = AllocateLayerWeights(K, H + 1, random);
        }

        public MultiLayerPerceptronModel(string fileName)
        {
            var input = new StreamReader(fileName);
            LoadClassLabels(input);
            W = LoadMatrix(input);
            _V = LoadMatrix(input);
            _activationFunction = LoadActivationFunction(input);
            input.Close();
        }
        
        /**
         * <summary> A constructor that takes {@link InstanceList}s as trainsSet and validationSet. It  sets the {@link NeuralNetworkModel}
         * nodes with given {@link InstanceList} then creates an input vector by using given trainSet and finds error.
         * Via the validationSet it finds the classification performance and reassigns the allocated weight Matrix with the matrix
         * that has the best accuracy and the Matrix V with the best Vector input.</summary>
         *
         * <param name="trainSet">     InstanceList that is used to train.</param>
         * <param name="validationSet">InstanceList that is used to validate.</param>
         * <param name="parameters">   Multi layer perceptron parameters; seed, learningRate, etaDecrease, crossValidationRatio, epoch, hiddenNodes.</param>
         */
        public MultiLayerPerceptronModel(InstanceList.InstanceList trainSet, InstanceList.InstanceList validationSet,
            MultiLayerPerceptronParameter parameters) : base(trainSet)
        {
            _activationFunction = parameters.GetActivationFunction();
            AllocateWeights(parameters.GetHiddenNodes(), new Random(parameters.GetSeed()));
            var bestW = (Matrix) W.Clone();
            var bestV = (Matrix) _V.Clone();
            var bestClassificationPerformance = new ClassificationPerformance(0.0);
            var epoch = parameters.GetEpoch();
            var learningRate = parameters.GetLearningRate();
            var activationDerivative = new Vector(1, 0.0);
            for (var i = 0; i < epoch; i++)
            {
                trainSet.Shuffle(parameters.GetSeed());
                for (var j = 0; j < trainSet.Size(); j++)
                {
                    CreateInputVector(trainSet.Get(j));
                    var hidden = CalculateHidden(x, W, _activationFunction);
                    var hiddenBiased = hidden.Biased();
                    var rMinusY = CalculateRMinusY(trainSet.Get(j), hiddenBiased, _V);
                    var deltaV = rMinusY.Multiply(hiddenBiased);
                    var tmph = _V.MultiplyWithVectorFromLeft(rMinusY);
                    tmph.Remove(0);
                    switch (_activationFunction)
                    {
                        case ActivationFunction.SIGMOID:
                            var oneMinusHidden = CalculateOneMinusHidden(hidden);
                            activationDerivative = oneMinusHidden.ElementProduct(hidden);
                            break;
                        case ActivationFunction.TANH:
                            var one = new Vector(hidden.Size(), 1.0);
                            hidden.Tanh();
                            activationDerivative = one.Difference(hidden.ElementProduct(hidden));
                            break;
                        case ActivationFunction.RELU:
                            hidden.ReluDerivative();
                            activationDerivative = hidden;
                            break;
                    }
                    var tmpHidden = tmph.ElementProduct(activationDerivative);
                    var deltaW = tmpHidden.Multiply(x);
                    deltaV.MultiplyWithConstant(learningRate);
                    _V.Add(deltaV);
                    deltaW.MultiplyWithConstant(learningRate);
                    W.Add(deltaW);
                }

                var currentClassificationPerformance = TestClassifier(validationSet);
                if (currentClassificationPerformance.GetAccuracy() > bestClassificationPerformance.GetAccuracy())
                {
                    bestClassificationPerformance = currentClassificationPerformance;
                    bestW = (Matrix) W.Clone();
                    bestV = (Matrix) _V.Clone();
                }

                learningRate *= parameters.GetEtaDecrease();
            }

            W = bestW;
            _V = bestV;
        }

        /**
         * <summary> The calculateOutput method calculates the forward single hidden layer by using Matrices W and V.</summary>
         */
        protected override void CalculateOutput()
        {
            CalculateForwardSingleHiddenLayer(W, _V, _activationFunction);
        }
    }
}