using Classification.Performance;
using Sampling;

namespace Classification.Experiment
{
    public class KFoldRun : MultipleRun
    {
        protected readonly int K;

        /**
         * <summary> Constructor for KFoldRun class. Basically sets K parameter of the K-fold cross-validation.</summary>
         *
         * <param name="k">K of the K-fold cross-validation.</param>
         */
        public KFoldRun(int k)
        {
            this.K = k;
        }

        protected void RunExperiment(Classifier.Classifier classifier, Parameter.Parameter parameter,
            ExperimentPerformance experimentPerformance, CrossValidation<Instance.Instance> crossValidation)
        {
            for (var i = 0; i < K; i++)
            {
                var trainSet = new InstanceList.InstanceList(crossValidation.GetTrainFold(i));
                var testSet = new InstanceList.InstanceList(crossValidation.GetTestFold(i));
                classifier.Train(trainSet, parameter);
                experimentPerformance.Add(classifier.Test(testSet));
            }
        }

        /**
         * <summary> Execute K-fold cross-validation with the given classifier on the given data set using the given parameters.</summary>
         *
         * <param name="experiment">Experiment to be run.</param>
         * <returns>An ExperimentPerformance instance.</returns>
         */
        public ExperimentPerformance Execute(Experiment experiment)
        {
            var result = new ExperimentPerformance();

            var crossValidation = new KFoldCrossValidation<Instance.Instance>(experiment.GetDataSet().GetInstances(),
                K, experiment.GetParameter().GetSeed());

            RunExperiment(experiment.GetClassifier(), experiment.GetParameter(), result, crossValidation);
            return result;
        }
    }
}