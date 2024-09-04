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

        /// <summary>
        /// Runs a K fold cross-validated experiment for the given model with the given parameters. Testing will be
        /// done on the separate test set. The experiment results will be added to the experimentPerformance.
        /// </summary>
        /// <param name="model">Classifier for the experiment</param>
        /// <param name="parameter">Hyperparameters of the model of the experiment</param>
        /// <param name="experimentPerformance">Storage to add experiment results</param>
        /// <param name="crossValidation">K-fold crossvalidated dataset.</param>
        /// <param name="testSet">Test set on which experiment performance is calculated.</param>
        protected void RunExperiment(Model.Model model, Parameter.Parameter parameter,
            ExperimentPerformance experimentPerformance, CrossValidation<Instance.Instance> crossValidation,
            InstanceList.InstanceList testSet)
        {
            for (var i = 0; i < k; i++)
            {
                var trainSet = new InstanceList.InstanceList(crossValidation.GetTrainFold(i));
                model.Train(trainSet, parameter);
                experimentPerformance.Add(model.Test(testSet));
            }
        }

        /**
         * Execute K-fold cross-validation with separate test set with the given classifier on the given data set using the given parameters.
         *
         * @param experiment Experiment to be run.
         * @return An ExperimentPerformance instance.
         */
        public override ExperimentPerformance Execute(Experiment experiment)
        {
            var result = new ExperimentPerformance();
            var instanceList = experiment.GetDataSet().GetInstanceList();
            var partition = instanceList.Partition(0.25, new Random(experiment.GetParameter().GetSeed()));
            var crossValidation = new KFoldCrossValidation<Instance.Instance>(
                partition.Get(1).GetInstances(),
                k, experiment.GetParameter().GetSeed());
            RunExperiment(experiment.GetModel(), experiment.GetParameter(), result, crossValidation,
                partition.Get(0));
            return result;
        }
    }
}