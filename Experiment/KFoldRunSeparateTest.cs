using System;
using Classification.Performance;
using Sampling;

namespace Classification.Experiment
{
    public class KFoldRunSeparateTest : KFoldRun
    {
        /**
         * Constructor for KFoldRunSeparateTest class. Basically sets K parameter of the K-fold cross-validation.
         *
         * @param K K of the K-fold cross-validation.
         */
        public KFoldRunSeparateTest(int k) : base(k)
        {
        }

        protected void RunExperiment(Classifier.Classifier classifier, Parameter.Parameter parameter,
            ExperimentPerformance experimentPerformance, CrossValidation<Instance.Instance> crossValidation,
            InstanceList.InstanceList testSet)
        {
            for (var i = 0; i < K; i++)
            {
                var trainSet = new InstanceList.InstanceList(crossValidation.GetTrainFold(i));
                classifier.Train(trainSet, parameter);
                experimentPerformance.Add(classifier.Test(testSet));
            }
        }

        /**
         * Execute K-fold cross-validation with separate test set with the given classifier on the given data set using the given parameters.
         *
         * @param experiment Experiment to be run.
         * @return An ExperimentPerformance instance.
         */
        public new ExperimentPerformance Execute(Experiment experiment)
        {
            var result = new ExperimentPerformance();
            var instanceList = experiment.GetDataSet().GetInstanceList();
            var partition = instanceList.Partition(0.25, new Random(experiment.GetParameter().GetSeed()));
            var crossValidation = new KFoldCrossValidation<Instance.Instance>(
                partition.Get(1).GetInstances(),
                K, experiment.GetParameter().GetSeed());
            RunExperiment(experiment.GetClassifier(), experiment.GetParameter(), result, crossValidation,
                partition.Get(0));
            return result;
        }
    }
}