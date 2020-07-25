using System;
using Classification.Model;
using Classification.Parameter;

namespace Classification.Classifier
{
    public class LinearPerceptron : Classifier
    {
        /**
         * <summary> Training algorithm for the linear perceptron algorithm. 20 percent of the data is separated as cross-validation
         * data used for selecting the best weights. 80 percent of the data is used for training the linear perceptron with
         * gradient descent.</summary>
         *
         * <param name="trainSet">  Training data given to the algorithm</param>
         * <param name="parameters">Parameters of the linear perceptron.</param>
         */
        public override void Train(InstanceList.InstanceList trainSet, Parameter.Parameter parameters)
        {
            var partition = trainSet.StratifiedPartition(
                ((LinearPerceptronParameter) parameters).GetCrossValidationRatio(), new Random(parameters.GetSeed()));
            model = new LinearPerceptronModel(partition.Get(1), partition.Get(0),
                (LinearPerceptronParameter) parameters);
        }
    }
}