using System;
using System.IO;
using Classification.Parameter;
using Classification.Performance;
using Math;

namespace Classification.Model
{
    public class LinearPerceptronModel : NeuralNetworkModel
    {
        protected Matrix W;

        /// <summary>
        /// Default constructor
        /// </summary>
        public LinearPerceptronModel()
        {
            
        }

        /// <summary>
        /// Loads a linear perceptron model from an input model file.
        /// </summary>
        /// <param name="fileName">Model file name.</param>
        private void Load(string fileName)
        {
            var input = new StreamReader(fileName);
            LoadClassLabels(input);
            W = LoadMatrix(input);
            input.Close();
        }
        
        /// <summary>
        /// Loads a linear perceptron model from an input model file.
        /// </summary>
        /// <param name="fileName">Model file name.</param>
        public LinearPerceptronModel(string fileName)
        {
            Load(fileName);
        }

        /**
         * <summary> The calculateOutput method calculates the {@link Matrix} y by multiplying Matrix W with {@link Vector} x.</summary>
         */
        protected override void CalculateOutput()
        {
            y = W.MultiplyWithVectorFromRight(x);
        }
        
        /**
         * <summary> Training algorithm for the linear perceptron algorithm. 20 percent of the data is separated as cross-validation
         * data used for selecting the best weights. 80 percent of the data is used for training the linear perceptron with
         * gradient descent.</summary>
         *
         * <param name="train">  Training data given to the algorithm</param>
         * <param name="_params">Parameters of the linear perceptron.</param>
         */
        public override void Train(InstanceList.InstanceList train, Parameter.Parameter _params)
        {
            Initialize(train);
            var parameters = (LinearPerceptronParameter) _params;
            var partition = train.StratifiedPartition(
                parameters.GetCrossValidationRatio(), new Random(parameters.GetSeed()));
            var trainSet = partition.Get(1);
            var validationSet = partition.Get(0);
            W = AllocateLayerWeights(K, d + 1, new Random(parameters.GetSeed()));
            var bestW = (Matrix) W.Clone();
            var bestClassificationPerformance = new ClassificationPerformance(0.0);
            var epoch = parameters.GetEpoch();
            var learningRate = parameters.GetLearningRate();
            for (var i = 0; i < epoch; i++)
            {
                trainSet.Shuffle(parameters.GetSeed());
                for (var j = 0; j < trainSet.Size(); j++)
                {
                    CreateInputVector(trainSet.Get(j));
                    var rMinusY = CalculateRMinusY(trainSet.Get(j), x, W);
                    var deltaW = rMinusY.Multiply(x);
                    deltaW.MultiplyWithConstant(learningRate);
                    W.Add(deltaW);
                }

                var currentClassificationPerformance = TestClassifier(validationSet);
                if (currentClassificationPerformance.GetAccuracy() > bestClassificationPerformance.GetAccuracy())
                {
                    bestClassificationPerformance = currentClassificationPerformance;
                    bestW = (Matrix) W.Clone();
                }

                learningRate *= parameters.GetEtaDecrease();
            }

            W = bestW;

        }

        /// <summary>
        /// Loads the linear perceptron model from an input file.
        /// </summary>
        /// <param name="fileName">File name of the linear perceptron model.</param>
        public override void LoadModel(string fileName)
        {
            Load(fileName);
        }

    }
}