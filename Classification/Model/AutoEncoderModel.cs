using System;
using System.Collections.Generic;
using Classification.Parameter;
using Math;

namespace Classification.Model
{
    public class AutoEncoderModel : NeuralNetworkModel
    {
        private Matrix _V, _W;

        /**
         * <summary> The allocateWeights method takes an integer number and sets layer weights of W and V matrices according to given number.</summary>
         *
         * <param name="H">Integer input.</param>
         * <param name="random">Random generator</param>
         */
        private void AllocateWeights(int H, Random random)
        {
            _W = AllocateLayerWeights(H, d + 1, random);
            _V = AllocateLayerWeights(K, H + 1, random);
        }

        /**
         * <summary> The testAutoEncoder method takes an {@link InstanceList} as an input and tries to predict a value and finds the difference with the
         * actual value for each item of that InstanceList. At the end, it returns an error rate by finding the mean of total errors.</summary>
         *
         * <param name="data">{@link InstanceList} to use as validation set.</param>
         * <returns>Error rate by finding the mean of total errors.</returns>
         */
        public Performance.Performance TestAutoEncoder(InstanceList.InstanceList data)
        {
            double total = data.Size();
            var error = 0.0;
            for (var i = 0; i < total; i++)
            {
                y = PredictInput(data.Get(i));
                r = data.Get(i).ToVector();
                error += r.Difference(y).DotProduct();
            }

            return new Performance.Performance(error / total);
        }

        /**
         * <summary> The predictInput method takes an {@link Instance} as an input and calculates a forward single hidden layer and returns the predicted value.</summary>
         *
         * <param name="instance">{@link Instance} to predict.</param>
         * <returns>Predicted value.</returns>
         */
        private Vector PredictInput(Instance.Instance instance)
        {
            CreateInputVector(instance);
            CalculateForwardSingleHiddenLayer(_W, _V, ActivationFunction.SIGMOID);
            return y;
        }

        /**
         * <summary> The calculateOutput method calculates a forward single hidden layer.</summary>
         */
        protected override void CalculateOutput()
        {
            CalculateForwardSingleHiddenLayer(_W, _V, ActivationFunction.SIGMOID);
        }

        public override Dictionary<string, double> PredictProbability(Instance.Instance instance)
        {
            return null;
        }

        /**
         * <summary> Training algorithm for auto encoders. An auto encoder is a neural network which attempts to replicate its input at its output.</summary>
         *
         * <param name="train">  Training data given to the algorithm.</param>
         * <param name="_params">Parameters of the auto encoder.</param>
         */
        public override void Train(InstanceList.InstanceList train, Parameter.Parameter _params)
        {
            Initialize(train);
            var parameters = (MultiLayerPerceptronParameter) _params;
            var partition = train.StratifiedPartition(0.2, new Random(_params.GetSeed()));
            var trainSet = partition.Get(1);
            var validationSet = partition.Get(0);
            AllocateWeights(parameters.GetHiddenNodes(), new Random(parameters.GetSeed()));
            var bestW = (Matrix)_W.Clone();
            var bestV = (Matrix)_V.Clone();
            var bestPerformance = new Performance.Performance(double.MaxValue);
            var epoch = parameters.GetEpoch();
            var learningRate = parameters.GetLearningRate();
            for (var i = 0; i < epoch; i++)
            {
                trainSet.Shuffle(parameters.GetSeed());
                for (var j = 0; j < trainSet.Size(); j++)
                {
                    CreateInputVector(trainSet.Get(j));
                    r = trainSet.Get(j).ToVector();
                    var hidden = CalculateHidden(x, _W, ActivationFunction.SIGMOID);
                    var hiddenBiased = hidden.Biased();
                    y = _V.MultiplyWithVectorFromRight(hiddenBiased);
                    var rMinusY = r.Difference(y);
                    var deltaV = rMinusY.Multiply(hiddenBiased);
                    var oneMinusHidden = CalculateOneMinusHidden(hidden);
                    var tmph = _V.MultiplyWithVectorFromLeft(rMinusY);
                    tmph.Remove(0);
                    var tmpHidden = oneMinusHidden.ElementProduct(hidden.ElementProduct(tmph));
                    var deltaW = tmpHidden.Multiply(x);
                    deltaV.MultiplyWithConstant(learningRate);
                    _V.Add(deltaV);
                    deltaW.MultiplyWithConstant(learningRate);
                    _W.Add(deltaW);
                }

                var currentPerformance = TestAutoEncoder(validationSet);
                if (currentPerformance.GetErrorRate() < bestPerformance.GetErrorRate())
                {
                    bestPerformance = currentPerformance;
                    bestW = (Matrix)_W.Clone();
                    bestV = (Matrix)_V.Clone();
                }

                learningRate *= 0.95;
            }

            _W = bestW;
            _V = bestV;
        }

        public override void LoadModel(string fileName)
        {
        }


        /**
         * <summary> A performance test for an auto encoder with the given test set..</summary>
         *
         * <param name="testSet">Test data (list of instances) to be tested.</param>
         * <returns>Error rate.</returns>
         */
        public override Performance.Performance Test(InstanceList.InstanceList testSet)
        {
            return TestAutoEncoder(testSet);
        }
    }
}