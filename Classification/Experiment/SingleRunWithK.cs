using Sampling;

namespace Classification.Experiment
{
    public class SingleRunWithK : SingleRun
    {
        private readonly int _k;

        /**
         * <summary> Constructor for SingleRunWithK class. Basically sets K parameter of the K-fold cross-validation.</summary>
         *
         * <param name="k">K of the K-fold cross-validation.</param>
         */
        public SingleRunWithK(int k)
        {
            this._k = k;
        }

        /// <summary>
        /// Runs first fold of a K fold cross-validated experiment for the given model with the given parameters.
        /// The experiment result will be returned.
        /// </summary>
        /// <param name="model">Classifier for the experiment</param>
        /// <param name="parameter">Hyperparameters of the model of the experiment</param>
        /// <param name="crossValidation">K-fold crossvalidated dataset.</param>
        /// <returns>The experiment result of the first fold of the K-fold cross-validated experiment.</returns>
        protected Performance.Performance RunExperiment(Model.Model model, Parameter.Parameter parameter,
            CrossValidation<Instance.Instance> crossValidation)
        {
            var trainSet = new InstanceList.InstanceList(crossValidation.GetTrainFold(0));
            var testSet = new InstanceList.InstanceList(crossValidation.GetTestFold(0));
            return model.SingleRun(parameter, trainSet, testSet);
        }


        /**
         * <summary> Execute Single K-fold cross-validation with the given classifier on the given data set using the given parameters.</summary>
         *
         * <param name="experiment">Experiment to be run.</param>
         * <returns>A Performance instance</returns>
         */
        public Performance.Performance Execute(Experiment experiment)
        {
            var crossValidation =
                new KFoldCrossValidation<Instance.Instance>(experiment.GetDataSet().GetInstances(), _k,
                    experiment.GetParameter().GetSeed());
            return RunExperiment(experiment.GetModel(), experiment.GetParameter(), crossValidation);
        }
    }
}