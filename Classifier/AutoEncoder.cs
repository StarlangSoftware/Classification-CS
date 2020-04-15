using System;
using Classification.InstanceList;
using Classification.Model;
using Classification.Parameter;

namespace Classification.Classifier
{
    public class AutoEncoder : Classifier
    {
        /**
         * <summary> Training algorithm for auto encoders. An auto encoder is a neural network which attempts to replicate its input at its output.</summary>
         *
         * <param name="trainSet">  Training data given to the algorithm.</param>
         * <param name="parameters">Parameters of the auto encoder.</param>
         */
        public override void Train(InstanceList.InstanceList trainSet, Parameter.Parameter parameters)
        {
            Partition partition = trainSet.StratifiedPartition(0.2, new Random(parameters.GetSeed()));
            model = new AutoEncoderModel(partition.Get(1), partition.Get(0),
                (MultiLayerPerceptronParameter) parameters);
        }


        /**
         * <summary> A performance test for an auto encoder with the given test set..</summary>
         *
         * <param name="testSet">Test data (list of instances) to be tested.</param>
         * <returns>Error rate.</returns>
         */
        public new Performance.Performance Test(InstanceList.InstanceList testSet)
        {
            return ((AutoEncoderModel) model).TestAutoEncoder(testSet);
        }
    }
}