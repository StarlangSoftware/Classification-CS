using Classification.Performance;
using Sampling;

namespace Classification.Experiment
{
    public class MxKFoldRun : KFoldRun
    {
        protected readonly int m;

        /**
         * <summary> Constructor for MxKFoldRun class. Basically sets K parameter of the K-fold cross-validation and M for the number of times.</summary>
         *
         * <param name="m">number of cross-validation times.</param>
         * <param name="k">K of the K-fold cross-validation.</param>
         */
        public MxKFoldRun(int m, int k) : base(k)
        {
            this.m = m;
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
            for (var j = 0; j < m; j++)
            {
                var crossValidation =
                    new KFoldCrossValidation<Instance.Instance>(experiment.GetDataSet().GetInstances(), k,
                        experiment.GetParameter().GetSeed());
                RunExperiment(experiment.GetModel(), experiment.GetParameter(), result, crossValidation);
            }

            return result;
        }
    }
}