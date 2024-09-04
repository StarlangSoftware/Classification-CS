using System;
using System.Collections.Generic;
using System.IO;
using Classification.Parameter;
using Classification.Performance;
using Math;

namespace Classification.Model
{
    public class DeepNetworkModel : NeuralNetworkModel
    {
        private List<Matrix> _weights;
        private int _hiddenLayerSize;
        private ActivationFunction _activationFunction;

        /**
         * <summary> The allocateWeights method takes {@link DeepNetworkParameter}s as an input. First it adds random weights to the {@link List}
         * of {@link Matrix} weights' first layer. Then it loops through the layers and adds random weights till the last layer.
         * At the end it adds random weights to the last layer and also sets the hiddenLayerSize value.</summary>
         *
         * <param name="parameters">{@link DeepNetworkParameter} input.</param>
         */
        private void AllocateWeights(DeepNetworkParameter parameters)
        {
            _weights = new List<Matrix>
                { AllocateLayerWeights(parameters.GetHiddenNodes(0), d + 1, new Random(parameters.GetSeed())) };
            for (var i = 0; i < parameters.LayerSize() - 1; i++)
            {
                _weights.Add(AllocateLayerWeights(parameters.GetHiddenNodes(i + 1), parameters.GetHiddenNodes(i) + 1,
                    new Random(parameters.GetSeed())));
            }

            _weights.Add(AllocateLayerWeights(K, parameters.GetHiddenNodes(parameters.LayerSize() - 1) + 1,
                new Random(parameters.GetSeed())));
            _hiddenLayerSize = parameters.LayerSize();
        }

        /**
         * <summary> The setBestWeights method creates an {@link List} of Matrix as bestWeights and clones the values of weights {@link List}
         * into this newly created {@link List}.</summary>
         *
         * <returns>An {@link List} clones from the weights List.</returns>
         */
        private List<Matrix> SetBestWeights()
        {
            var bestWeights = new List<Matrix>();
            foreach (var m in _weights)
            {
                bestWeights.Add((Matrix)m.Clone());
            }

            return bestWeights;
        }

        public DeepNetworkModel()
        {
        }

        /// <summary>
        /// Loads a deep network model from an input model file.
        /// </summary>
        /// <param name="fileName">Model file name.</param>
        private void Load(string fileName)
        {
            var input = new StreamReader(fileName);
            LoadClassLabels(input);
            _hiddenLayerSize = int.Parse(input.ReadLine());
            _weights = new List<Matrix>();
            for (var i = 0; i < _hiddenLayerSize + 1; i++)
            {
                _weights.Add(LoadMatrix(input));
            }

            _activationFunction = LoadActivationFunction(input);
            input.Close();
        }

        /// <summary>
        /// Loads a deep network model from an input model file.
        /// </summary>
        /// <param name="fileName">Model file name.</param>
        public DeepNetworkModel(string fileName)
        {
            Load(fileName);
        }

        /**
         * <summary> The calculateOutput method loops size of the weights times and calculate one hidden layer at a time and adds bias term.
         * At the end it updates the output y value.</summary>
         */
        protected override void CalculateOutput()
        {
            Vector hiddenBiased = null;
            for (var i = 0; i < _weights.Count - 1; i++)
            {
                Vector hidden;
                if (i == 0)
                {
                    hidden = CalculateHidden(x, _weights[i], _activationFunction);
                }
                else
                {
                    hidden = CalculateHidden(hiddenBiased, _weights[i], _activationFunction);
                }

                hiddenBiased = hidden.Biased();
            }

            y = _weights[_weights.Count - 1].MultiplyWithVectorFromRight(hiddenBiased);
        }

        /**
         * <summary> Training algorithm for deep network classifier.</summary>
         *
         * <param name="train">  Training data given to the algorithm.</param>
         * <param name="_params">Parameters of the deep network algorithm. crossValidationRatio and seed are used as parameters.</param>
         */
        public override void Train(InstanceList.InstanceList train, Parameter.Parameter _params)
        {
            Initialize(train);
            var parameters = ((DeepNetworkParameter)_params);
            var partition =
                train.StratifiedPartition(parameters.GetCrossValidationRatio(), new Random(parameters.GetSeed()));
            var trainSet = partition.Get(1);
            var validationSet = partition.Get(0);
            var deltaWeights = new List<Matrix>();
            var hidden = new List<Vector>();
            var hiddenBiased = new List<Vector>();
            _activationFunction = parameters.GetActivationFunction();
            AllocateWeights(parameters);
            var bestWeights = SetBestWeights();
            var bestClassificationPerformance = new ClassificationPerformance(0.0);
            var epoch = parameters.GetEpoch();
            var learningRate = parameters.GetLearningRate();
            Vector tmph;
            var tmpHidden = new Vector(1, 0.0);
            var activationDerivative = new Vector(1, 0.0);
            for (var i = 0; i < epoch; i++)
            {
                trainSet.Shuffle(parameters.GetSeed());
                for (var j = 0; j < trainSet.Size(); j++)
                {
                    CreateInputVector(trainSet.Get(j));
                    hidden.Clear();
                    hiddenBiased.Clear();
                    deltaWeights.Clear();
                    for (var k = 0; k < _hiddenLayerSize; k++)
                    {
                        if (k == 0)
                        {
                            hidden.Add(CalculateHidden(x, _weights[k], _activationFunction));
                        }
                        else
                        {
                            hidden.Add(CalculateHidden(hiddenBiased[k - 1], _weights[k], _activationFunction));
                        }

                        hiddenBiased.Add(hidden[k].Biased());
                    }

                    var rMinusY = CalculateRMinusY(trainSet.Get(j), hiddenBiased[_hiddenLayerSize - 1],
                        _weights[_weights.Count - 1]);
                    deltaWeights.Insert(0, rMinusY.Multiply(hiddenBiased[_hiddenLayerSize - 1]));
                    for (var k = _weights.Count - 2; k >= 0; k--)
                    {
                        if (k == _weights.Count - 2)
                        {
                            tmph = _weights[k + 1].MultiplyWithVectorFromLeft(rMinusY);
                        }
                        else
                        {
                            tmph = _weights[k + 1].MultiplyWithVectorFromLeft(tmpHidden);
                        }

                        tmph.Remove(0);
                        switch (_activationFunction)
                        {
                            case ActivationFunction.SIGMOID:
                                var oneMinusHidden = CalculateOneMinusHidden(hidden[k]);
                                activationDerivative = oneMinusHidden.ElementProduct(hidden[k]);
                                break;
                            case ActivationFunction.TANH:
                                var one = new Vector(hidden.Count, 1.0);
                                hidden[k].Tanh();
                                activationDerivative = one.Difference(hidden[k].ElementProduct(hidden[k]));
                                break;
                            case ActivationFunction.RELU:
                                hidden[k].ReluDerivative();
                                activationDerivative = hidden[k];
                                break;
                        }

                        tmpHidden = tmph.ElementProduct(activationDerivative);
                        if (k == 0)
                        {
                            deltaWeights.Insert(0, tmpHidden.Multiply(x));
                        }
                        else
                        {
                            deltaWeights.Insert(0, tmpHidden.Multiply(hiddenBiased[k - 1]));
                        }
                    }

                    for (var k = 0; k < _weights.Count; k++)
                    {
                        deltaWeights[k].MultiplyWithConstant(learningRate);
                        _weights[k].Add(deltaWeights[k]);
                    }
                }

                var currentClassificationPerformance = TestClassifier(validationSet);
                if (currentClassificationPerformance.GetAccuracy() > bestClassificationPerformance.GetAccuracy())
                {
                    bestClassificationPerformance = currentClassificationPerformance;
                    bestWeights = SetBestWeights();
                }

                learningRate *= parameters.GetEtaDecrease();
            }

            _weights.Clear();
            foreach (var m in bestWeights)
            {
                _weights.Add(m);
            }
        }

        /// <summary>
        /// Loads the deep network model from an input file.
        /// </summary>
        /// <param name="fileName">File name of the deep network model.</param>
        public override void LoadModel(string fileName)
        {
            Load(fileName);
        }
    }
}