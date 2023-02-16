using Classification.Classifier;
using Classification.DistanceMetric;
using Classification.Experiment;
using Classification.Parameter;
using Classification.Performance;
using NUnit.Framework;
using Test.Classifier;

namespace Test.Experiment
{
    public class KFoldRunTest : ClassifierTest
    {
        [Test]
        public void TestExecute()
        {
            var kFoldRun = new KFoldRun(10);
            var experimentPerformance =
                kFoldRun.Execute(new Classification.Experiment.Experiment(new C45(), new C45Parameter(1, true, 0.2), iris));
            Assert.AreEqual(6.67, 100 * experimentPerformance.MeanPerformance().GetErrorRate(), 0.01);
            experimentPerformance =
                kFoldRun.Execute(new Classification.Experiment.Experiment(new C45(), new C45Parameter(1, true, 0.2), tictactoe));
            Assert.AreEqual(17.22, 100 * experimentPerformance.MeanPerformance().GetErrorRate(), 0.01);
            experimentPerformance =
                kFoldRun.Execute(new Classification.Experiment.Experiment(new Knn(), new KnnParameter(1, 3, new EuclidianDistance()), bupa));
            Assert.AreEqual(37.44, 100 * experimentPerformance.MeanPerformance().GetErrorRate(), 0.01);
            experimentPerformance = kFoldRun.Execute(new Classification.Experiment.Experiment(new Knn(),
                new KnnParameter(1, 3, new EuclidianDistance()), dermatology));
            Assert.AreEqual(9.59, 100 * experimentPerformance.MeanPerformance().GetErrorRate(), 0.01);
            experimentPerformance = kFoldRun.Execute(new Classification.Experiment.Experiment(new Lda(), new Parameter(1), bupa));
            Assert.AreEqual(31.83, 100 * experimentPerformance.MeanPerformance().GetErrorRate(), 0.01);
            experimentPerformance = kFoldRun.Execute(new Classification.Experiment.Experiment(new Lda(), new Parameter(1), dermatology));
            Assert.AreEqual(2.18, 100 * experimentPerformance.MeanPerformance().GetErrorRate(), 0.01);
            experimentPerformance = kFoldRun.Execute(new Classification.Experiment.Experiment(new LinearPerceptron(),
                new LinearPerceptronParameter(1, 0.1, 0.99, 0.2, 100), iris));
            Assert.AreEqual(3.33, 100 * experimentPerformance.MeanPerformance().GetErrorRate(), 0.01);
            experimentPerformance = kFoldRun.Execute(new Classification.Experiment.Experiment(new LinearPerceptron(),
                new LinearPerceptronParameter(1, 0.1, 0.99, 0.2, 100), dermatology));
            Assert.AreEqual(3.54, 100 * experimentPerformance.MeanPerformance().GetErrorRate(), 0.01);
            experimentPerformance = kFoldRun.Execute(new Classification.Experiment.Experiment(new NaiveBayes(), new Parameter(1), car));
            Assert.AreEqual(14.64, 100 * experimentPerformance.MeanPerformance().GetErrorRate(), 0.01);
            experimentPerformance = kFoldRun.Execute(new Classification.Experiment.Experiment(new NaiveBayes(), new Parameter(1), nursery));
            Assert.AreEqual(9.71, 100 * experimentPerformance.MeanPerformance().GetErrorRate(), 0.01);
            experimentPerformance =
                kFoldRun.Execute(new Classification.Experiment.Experiment(new Bagging(), new BaggingParameter(1, 50), tictactoe));
            Assert.AreEqual(3.03, 100 * experimentPerformance.MeanPerformance().GetErrorRate(), 0.01);
            experimentPerformance = kFoldRun.Execute(new Classification.Experiment.Experiment(new Bagging(), new BaggingParameter(1, 50), car));
            Assert.AreEqual(6.25, 100 * experimentPerformance.MeanPerformance().GetErrorRate(), 0.01);
            experimentPerformance = kFoldRun.Execute(new Classification.Experiment.Experiment(new Dummy(), new Parameter(1), nursery));
            Assert.AreEqual(67.17, 100 * experimentPerformance.MeanPerformance().GetErrorRate(), 0.01);
            experimentPerformance = kFoldRun.Execute(new Classification.Experiment.Experiment(new Dummy(), new Parameter(1), iris));
            Assert.AreEqual(80.00, 100 * experimentPerformance.MeanPerformance().GetErrorRate(), 0.01);
        }
    }
}