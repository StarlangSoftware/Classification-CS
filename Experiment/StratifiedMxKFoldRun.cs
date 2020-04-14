using Classification.Performance;
using Sampling;

namespace Classification.Experiment
{
    public class StratifiedMxKFoldRun : MxKFoldRun
    {
        /**
         * <summary> Constructor for StratifiedMxKFoldRun class. Basically sets K parameter of the K-fold cross-validation and M for the number of times.</summary>
         *
         * <param name="m">number of cross-validation times.</param>
         * <param name="k">K of the K-fold cross-validation.</param>
         */
        public StratifiedMxKFoldRun(int m, int k) : base(m, k)
        {
        }

        /**
         * <summary> Execute the Stratified MxK-fold cross-validation with the given classifier on the given data set using the given parameters.</summary>
         *
         * <param name="experiment">Experiment to be run.</param>
         * <returns>An ExperimentPerformance instance.</returns>
         */
        public new ExperimentPerformance Execute(Experiment experiment)
        {
            var result = new ExperimentPerformance();
            for (var j = 0; j < M; j++)
            {
                var crossValidation =
                    new StratifiedKFoldCrossValidation<Instance.Instance>(experiment.GetDataSet().GetClassInstances(),
                        K,
                        experiment.GetParameter().GetSeed());
                RunExperiment(experiment.GetClassifier(), experiment.GetParameter(), result, crossValidation);
            }

            return result;
        }
    }
}