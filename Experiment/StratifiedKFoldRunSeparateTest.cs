using System;
using Classification.Performance;
using Sampling;

namespace Classification.Experiment
{
    public class StratifiedKFoldRunSeparateTest : KFoldRunSeparateTest
    {
        /**
         * <summary> Constructor for StratifiedKFoldRunSeparateTest class. Basically sets K parameter of the K-fold cross-validation.</summary>
         *
         * <param name="k">K of the K-fold cross-validation.</param>
         */
        public StratifiedKFoldRunSeparateTest(int k) : base(k)
        {
        }

        /**
         * <summary> Execute Stratified K-fold cross-validation with the given classifier on the given data set using the given parameters.</summary>
         *
         * <param name="experiment">Experiment to be run.</param>
         * <returns>An ExperimentPerformance instance.</returns>
         */
        public new ExperimentPerformance Execute(Experiment experiment)
        {
            var result = new ExperimentPerformance();
            var instanceList = experiment.GetDataSet().GetInstanceList();
            var partition = instanceList.Partition(0.25, new Random(experiment.GetParameter().GetSeed()));
            var crossValidation =
                new StratifiedKFoldCrossValidation<Instance.Instance>(partition.Get(1).DivideIntoClasses().GetLists(),
                    K, experiment.GetParameter().GetSeed());
            RunExperiment(experiment.GetClassifier(), experiment.GetParameter(), result, crossValidation,
                partition.Get(0));
            return result;
        }
    }
}