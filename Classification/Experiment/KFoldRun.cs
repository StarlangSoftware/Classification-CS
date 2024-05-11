using Classification.Performance;
using Sampling;

namespace Classification.Experiment
{
    public class KFoldRun : MultipleRun
    {
        protected readonly int k;

        /**
         * <summary> Constructor for KFoldRun class. Basically sets K parameter of the K-fold cross-validation.</summary>
         *
         * <param name="k">K of the K-fold cross-validation.</param>
         */
        public KFoldRun(int k)
        {
            this.k = k;
        }

        /// <summary>
        /// Runs a K fold cross-validated experiment for the given classifier with the given parameters. The experiment
        /// results will be added to the experimentPerformance.
        /// </summary>
        /// <param name="classifier">Classifier for the experiment</param>
        /// <param name="parameter">Hyperparameters of the classifier of the experiment</param>
        /// <param name="experimentPerformance">Storage to add experiment results</param>
        /// <param name="crossValidation">K-fold crossvalidated dataset.</param>
        protected void RunExperiment(Classifier.Classifier classifier, Parameter.Parameter parameter,
            ExperimentPerformance experimentPerformance, CrossValidation<Instance.Instance> crossValidation)
        {
            for (var i = 0; i < k; i++)
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
        public virtual ExperimentPerformance Execute(Experiment experiment)
        {
            var result = new ExperimentPerformance();

            var crossValidation = new KFoldCrossValidation<Instance.Instance>(experiment.GetDataSet().GetInstances(),
                k, experiment.GetParameter().GetSeed());

            RunExperiment(experiment.GetClassifier(), experiment.GetParameter(), result, crossValidation);
            return result;
        }
    }
}