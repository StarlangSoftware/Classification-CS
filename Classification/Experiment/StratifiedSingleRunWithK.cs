using Sampling;

namespace Classification.Experiment
{
    public class StratifiedSingleRunWithK
    {
        private readonly int _k;

        /**
         * <summary> Constructor for StratifiedSingleRunWithK class. Basically sets K parameter of the K-fold cross-validation.</summary>
         *
         * <param name="k">K of the K-fold cross-validation.</param>
         */
        public StratifiedSingleRunWithK(int k)
        {
            this._k = k;
        }

        /**
         * <summary> Execute Stratified Single K-fold cross-validation with the given classifier on the given data set using the given parameters.</summary>
         *
         * <param name="experiment">Experiment to be run.</param>
         * <returns>A Performance instance.</returns>
         */
        public Performance.Performance Execute(Experiment experiment)
        {
            var crossValidation =
                new StratifiedKFoldCrossValidation<Instance.Instance>(experiment.GetDataSet().GetClassInstances(), _k,
                    experiment.GetParameter().GetSeed());
            var trainSet = new InstanceList.InstanceList(crossValidation.GetTrainFold(0));
            var testSet = new InstanceList.InstanceList(crossValidation.GetTestFold(0));
            return experiment.GetClassifier().SingleRun(experiment.GetParameter(), trainSet, testSet);
        }
    }
}