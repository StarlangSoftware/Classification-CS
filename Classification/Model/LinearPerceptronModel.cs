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

        /**
         * <summary> Constructor that sets the {@link NeuralNetworkModel} nodes with given {@link InstanceList}.</summary>
         *
         * <param name="trainSet">InstanceList that is used to train.</param>
         */
        public LinearPerceptronModel(InstanceList.InstanceList trainSet) : base(trainSet)
        {
        }

        public LinearPerceptronModel()
        {
            
        }
        public LinearPerceptronModel(string fileName)
        {
            var input = new StreamReader(fileName);
            LoadClassLabels(input);
            W = LoadMatrix(input);
            input.Close();
        }
        /**
         * <summary> Constructor that takes {@link InstanceList}s as trainsSet and validationSet. Initially it allocates layer weights,
         * then creates an input vector by using given trainSet and finds error. Via the validationSet it finds the classification
         * performance and at the end it reassigns the allocated weight Matrix with the matrix that has the best accuracy.</summary>
         *
         * <param name="trainSet">     InstanceList that is used to train.</param>
         * <param name="validationSet">InstanceList that is used to validate.</param>
         * <param name="parameters">   Linear perceptron parameters; learningRate, etaDecrease, crossValidationRatio, epoch.</param>
         */
        public LinearPerceptronModel(InstanceList.InstanceList trainSet, InstanceList.InstanceList validationSet,
            LinearPerceptronParameter parameters) : base(trainSet)
        {
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

        /**
         * <summary> The calculateOutput method calculates the {@link Matrix} y by multiplying Matrix W with {@link Vector} x.</summary>
         */
        protected override void CalculateOutput()
        {
            y = W.MultiplyWithVectorFromRight(x);
        }
    }
}