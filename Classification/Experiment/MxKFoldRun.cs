using Classification.Performance;
using Sampling;

namespace Classification.Experiment
{
    public class MxKFoldRun : KFoldRun
    {
        protected readonly int M;

        /**
         * <summary> Constructor for MxKFoldRun class. Basically sets K parameter of the K-fold cross-validation and M for the number of times.</summary>
         *
         * <param name="m">number of cross-validation times.</param>
         * <param name="k">K of the K-fold cross-validation.</param>
         */
        public MxKFoldRun(int m, int k) : base(k)
        {
            this.M = m;
        }

        /**
         * <summary> Execute the MxKFold run with the given classifier on the given data set using the given parameters.</summary>
         *
         * <param name="experiment">Experiment to be run.</param>
         * <returns>An ExperimentPerformance instance.</returns>
         */
        public override ExperimentPerformance Execute(Experiment experiment)
        {
            var result = new ExperimentPerformance();
            for (var j = 0; j < M; j++)
            {
                var crossValidation =
                    new KFoldCrossValidation<Instance.Instance>(experiment.GetDataSet().GetInstances(), K,
                        experiment.GetParameter().GetSeed());
                RunExperiment(experiment.GetClassifier(), experiment.GetParameter(), result, crossValidation);
            }

            return result;
        }
    }
}