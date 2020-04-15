using System;
using Classification.Model;
using Classification.Parameter;

namespace Classification.Classifier
{
    public class DeepNetwork : Classifier
    {
        /**
         * <summary> Training algorithm for deep network classifier.</summary>
         *
         * <param name="trainSet">  Training data given to the algorithm.</param>
         * <param name="parameters">Parameters of the deep network algorithm. crossValidationRatio and seed are used as parameters.</param>
         */
        public override void Train(InstanceList.InstanceList trainSet, Parameter.Parameter parameters)
        {
            var partition = trainSet.StratifiedPartition(
                ((DeepNetworkParameter) parameters).GetCrossValidationRatio(), new Random(parameters.GetSeed()));
            model = new DeepNetworkModel(partition.Get(1), partition.Get(0), (DeepNetworkParameter) parameters);
        }
    }
}