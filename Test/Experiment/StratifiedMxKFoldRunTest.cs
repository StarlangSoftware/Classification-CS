using Classification.Classifier;
using Classification.DistanceMetric;
using Classification.Experiment;
using Classification.Parameter;
using Classification.Performance;
using NUnit.Framework;
using Test.Classifier;

namespace Test.Experiment
{
    public class StratifiedMxKFoldRunTest : ClassifierTest
    {
        [Test]
        public void TestExecute()
        {
            var stratifiedMxKFoldRun = new StratifiedMxKFoldRun(5, 2);

            var experimentPerformance =
                stratifiedMxKFoldRun.Execute(new Classification.Experiment.Experiment(new C45(), new C45Parameter(1, true, 0.2), iris));

            Assert.AreEqual(6.67, 100 * experimentPerformance.MeanPerformance().GetErrorRate(), 0.01);
            experimentPerformance = stratifiedMxKFoldRun.Execute(new Classification.Experiment.Experiment(new C45(), new
                C45Parameter(1, true, 0.2), tictactoe));
            Assert.AreEqual(31.73, 100 * experimentPerformance.MeanPerformance().GetErrorRate(), 0.01);
            experimentPerformance = stratifiedMxKFoldRun.Execute(new Classification.Experiment.Experiment(new Knn(), new KnnParameter(1, 3, new
                EuclidianDistance()), bupa));
            Assert.AreEqual(35.06, 100 * experimentPerformance.MeanPerformance().GetErrorRate(), 0.01);
            experimentPerformance = stratifiedMxKFoldRun.Execute(new Classification.Experiment.Experiment(new Knn(), new KnnParameter(1, 3, new
                EuclidianDistance()), dermatology));
            Assert.AreEqual(15.05, 100 * experimentPerformance.MeanPerformance().GetErrorRate(), 0.01);
            experimentPerformance = stratifiedMxKFoldRun.Execute(new Classification.Experiment.Experiment(new Lda(), new Parameter(1), bupa));
            Assert.AreEqual(28.98, 100 * experimentPerformance.MeanPerformance().GetErrorRate(), 0.01);
            experimentPerformance =
                stratifiedMxKFoldRun.Execute(new Classification.Experiment.Experiment(new Lda(), new Parameter(1), dermatology));
            Assert.AreEqual(5.20, 100 * experimentPerformance.MeanPerformance().GetErrorRate(), 0.01);
            experimentPerformance = stratifiedMxKFoldRun.Execute(new Classification.Experiment.Experiment(new LinearPerceptron(), new
                LinearPerceptronParameter(1, 0.1, 0.99, 0.2, 100), iris));
            Assert.AreEqual(23.33, 100 * experimentPerformance.MeanPerformance().GetErrorRate(), 0.01);
            experimentPerformance = stratifiedMxKFoldRun.Execute(new Classification.Experiment.Experiment(new LinearPerceptron(), new
                LinearPerceptronParameter(1, 0.1, 0.99, 0.2, 100), dermatology));
            Assert.AreEqual(27.63, 100 * experimentPerformance.MeanPerformance().GetErrorRate(), 0.01);
            experimentPerformance =
                stratifiedMxKFoldRun.Execute(new Classification.Experiment.Experiment(new NaiveBayes(), new Parameter(1), car));
            Assert.AreEqual(46.99, 100 * experimentPerformance.MeanPerformance().GetErrorRate(), 0.01);
            experimentPerformance = stratifiedMxKFoldRun.Execute(
                new Classification.Experiment.Experiment(new NaiveBayes(), new Parameter(1), nursery)
            );
            Assert.AreEqual(63.29, 100 * experimentPerformance.MeanPerformance().GetErrorRate(), 0.01);
            experimentPerformance = stratifiedMxKFoldRun.Execute(new Classification.Experiment.Experiment(new Bagging(), new
                BaggingParameter(1, 50), tictactoe));
            Assert.AreEqual(42.48, 100 * experimentPerformance.MeanPerformance().GetErrorRate(), 0.01);
            experimentPerformance = stratifiedMxKFoldRun.Execute(new Classification.Experiment.Experiment(new Bagging(), new
                BaggingParameter(1, 50), car));
            Assert.AreEqual(61.19, 100 * experimentPerformance.MeanPerformance().GetErrorRate(), 0.01);
            experimentPerformance =
                stratifiedMxKFoldRun.Execute(new Classification.Experiment.Experiment(new Dummy(), new Parameter(1), nursery));
            Assert.AreEqual(66.67, 100 * experimentPerformance.MeanPerformance().GetErrorRate(), 0.01);
            experimentPerformance = stratifiedMxKFoldRun.Execute(new Classification.Experiment.Experiment(new Dummy(), new Parameter(1), iris));
            Assert.AreEqual(66.67, 100 * experimentPerformance.MeanPerformance().GetErrorRate(), 0.01);
        }
    }
}