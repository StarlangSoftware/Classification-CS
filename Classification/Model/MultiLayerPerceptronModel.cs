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
         * <param name="random">Random number generator</param>
         */
        private void AllocateWeights(int H, Random random)
        {
            W = AllocateLayerWeights(H, d + 1, random);
            _V = AllocateLayerWeights(K, H + 1, random);
        }

        /// <summary>
        /// Loads a multi-layer perceptron model from an input model file.
        /// </summary>
        /// <param name="fileName">Model file name.</param>
        private void Load(string fileName)
        {
            var input = new StreamReader(fileName);
            LoadClassLabels(input);
            W = LoadMatrix(input);
            _V = LoadMatrix(input);
            _activationFunction = LoadActivationFunction(input);
            input.Close();
        }

        /// <summary>
        /// Loads a multi-layer perceptron model from an input model file.
        /// </summary>
        /// <param name="fileName">Model file name.</param>
        public MultiLayerPerceptronModel(string fileName)
        {
            Load(fileName);
        }

        public MultiLayerPerceptronModel()
        {
        }

        /**
         * <summary> The calculateOutput method calculates the forward single hidden layer by using Matrices W and V.</summary>
         */
        protected override void CalculateOutput()
        {
            CalculateForwardSingleHiddenLayer(W, _V, _activationFunction);
        }

        /**
         * <summary> Training algorithm for the multilayer perceptron algorithm. 20 percent of the data is separated as cross-validation
         * data used for selecting the best weights. 80 percent of the data is used for training the multilayer perceptron with
         * gradient descent.</summary>
         *
         * <param name="train">  Training data given to the algorithm</param>
         * <param name="_params">Parameters of the multilayer perceptron.</param>
         */
        public override void Train(InstanceList.InstanceList train, Parameter.Parameter _params)
        {
            Initialize(train);
            var parameters = (MultiLayerPerceptronParameter)_params;
            var partition =
                train.StratifiedPartition(parameters.GetCrossValidationRatio(), new Random(parameters.GetSeed()));
            var trainSet = partition.Get(1);
            var validationSet = partition.Get(0);
            _activationFunction = parameters.GetActivationFunction();
            AllocateWeights(parameters.GetHiddenNodes(), new Random(parameters.GetSeed()));
            var bestW = (Matrix)W.Clone();
            var bestV = (Matrix)_V.Clone();
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
                    bestW = (Matrix)W.Clone();
                    bestV = (Matrix)_V.Clone();
                }

                learningRate *= parameters.GetEtaDecrease();
            }

            W = bestW;
            _V = bestV;
        }

        /// <summary>
        /// Loads the multi-layer perceptron model from an input file.
        /// </summary>
        /// <param name="fileName">File name of the multi-layer perceptron model.</param>
        public override void LoadModel(string fileName)
        {
            Load(fileName);
        }
    }
}