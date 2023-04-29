using System;
using Classification.Model;
using Classification.Parameter;

namespace Classification.Classifier
{
    public class MultiLayerPerceptron : Classifier
    {
        /**
         * <summary> Training algorithm for the multilayer perceptron algorithm. 20 percent of the data is separated as cross-validation
         * data used for selecting the best weights. 80 percent of the data is used for training the multilayer perceptron with
         * gradient descent.</summary>
         *
         * <param name="trainSet">  Training data given to the algorithm</param>
         * <param name="parameters">Parameters of the multilayer perceptron.</param>
         */
        public override void Train(InstanceList.InstanceList trainSet, Parameter.Parameter parameters)
        {
            var partition = trainSet.StratifiedPartition(
                ((MultiLayerPerceptronParameter) parameters).GetCrossValidationRatio(),
                new Random(parameters.GetSeed()));

            model = new MultiLayerPerceptronModel(partition.Get(1), partition.Get(0), (MultiLayerPerceptronParameter)
                parameters);
        }

        public override void LoadModel(string fileName)
        {
            model = new MultiLayerPerceptronModel(fileName);
        }
    }
}