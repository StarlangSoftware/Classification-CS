using System;
using Classification.Performance;
using Sampling;

namespace Classification.Experiment
{
    public class StratifiedMxKFoldRunSeparateTest : StratifiedKFoldRunSeparateTest
    {
        private readonly int _m;

        /**
         * Constructor for StratifiedMxKFoldRunSeparateTest class. Basically sets K parameter of the K-fold cross-validation and M for the number of times.
         *
         * @param M number of cross-validation times.
         * @param K K of the K-fold cross-validation.
         */
        public StratifiedMxKFoldRunSeparateTest(int m, int k) : base(k)
        {
            this._m = m;
        }

        /**
         * Execute the Stratified MxK-fold cross-validation with the given classifier on the given data set using the given parameters.
         *
         * @param experiment Experiment to be run.
         * @return An ExperimentPerformance instance.
         */
        public override ExperimentPerformance Execute(Experiment experiment)
        {
            var result = new ExperimentPerformance();
            for (var j = 0; j < _m; j++)
            {
                var instanceList = experiment.GetDataSet().GetInstanceList();
                var partition = instanceList.Partition(0.25, new Random(experiment.GetParameter().GetSeed()));
                var crossValidation =
                    new StratifiedKFoldCrossValidation<Instance.Instance>(partition.Get(1).DivideIntoClasses().GetLists(), K,
                        experiment.GetParameter().GetSeed());
                RunExperiment(experiment.GetClassifier(), experiment.GetParameter(), result, crossValidation,
                    partition.Get(0));
            }

            return result;
        }
    }
}