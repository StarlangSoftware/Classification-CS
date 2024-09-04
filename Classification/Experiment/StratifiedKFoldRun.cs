using Classification.Performance;
using Sampling;

namespace Classification.Experiment
{
    public class StratifiedKFoldRun : KFoldRun
    {
        /**
         * <summary> Constructor for KFoldRun class. Basically sets K parameter of the K-fold cross-validation.</summary>
         * <param name="k">K of the K-fold cross-validation.</param>
         */
        public StratifiedKFoldRun(int k) : base(k)
        {
        }

        /**
         * <summary> Execute Stratified K-fold cross-validation with the given classifier on the given data set using the given parameters.</summary>
         *
         * <param name="experiment">Experiment to be run.</param>
         * <returns>An ExperimentPerformance instance.</returns>
         */
        public override ExperimentPerformance Execute(Experiment experiment)
        {
            var result = new ExperimentPerformance();
            var crossValidation =
                new StratifiedKFoldCrossValidation<Instance.Instance>(experiment.GetDataSet().GetClassInstances(), k,
                    experiment.GetParameter().GetSeed());
            RunExperiment(experiment.GetModel(), experiment.GetParameter(), result, crossValidation);
            return result;
        }
    }
}