using Classification.Performance;
using Sampling;

namespace Classification.Experiment
{
    public class BootstrapRun : MultipleRun
    {
        private readonly int _numberOfBootstraps;

        /**
         * <summary> Constructor for BootstrapRun class. Basically sets the number of bootstrap runs.</summary>
         *
         * <param name="numberOfBootstraps">Number of bootstrap runs.</param>
         */
        public BootstrapRun(int numberOfBootstraps)
        {
            this._numberOfBootstraps = numberOfBootstraps;
        }

        /**
         * <summary> Execute the bootstrap run with the given classifier on the given data set using the given parameters.</summary>
         *
         * <param name="experiment">Experiment to be run.</param>
         * <returns>An ExperimentPerformance instance.</returns>
         */
        public ExperimentPerformance Execute(Experiment experiment)
        {
            var result = new ExperimentPerformance();
            for (var i = 0; i < _numberOfBootstraps; i++)
            {
                var bootstrap = new Bootstrap<Instance.Instance>(experiment.GetDataSet().GetInstances(),
                    i + experiment.GetParameter().GetSeed());
                var bootstrapSample = new InstanceList.InstanceList(bootstrap.GetSample());
                experiment.GetClassifier().Train(bootstrapSample, experiment.GetParameter());
                result.Add(experiment.GetClassifier().Test(experiment.GetDataSet().GetInstanceList()));
            }

            return result;
        }
    }
}