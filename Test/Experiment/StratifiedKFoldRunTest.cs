using Classification.DistanceMetric;
using Classification.Experiment;
using Classification.Model;
using Classification.Model.DecisionTree;
using Classification.Parameter;
using NUnit.Framework;
using Test.Classifier;

namespace Test.Experiment
{
    public class StratifiedKFoldRunTest : ClassifierTest
    {
        [Test]
        public void TestExecute()
        {
            var stratifiedKFoldRun = new StratifiedKFoldRun(10);
            var experimentPerformance =
                stratifiedKFoldRun.Execute(new Classification.Experiment.Experiment(new DecisionTree(), new C45Parameter(1, true, 0.2), iris));
            Assert.AreEqual(6.67, 100 * experimentPerformance.MeanPerformance().GetErrorRate(), 0.01);
            experimentPerformance =
                stratifiedKFoldRun.Execute(new Classification.Experiment.Experiment(new DecisionTree(), new C45Parameter(1, true, 0.2), tictactoe));
            Assert.AreEqual(20.05, 100 * experimentPerformance.MeanPerformance().GetErrorRate(), 0.01);
            experimentPerformance =
                stratifiedKFoldRun.Execute(new Classification.Experiment.Experiment(new KnnModel(), new KnnParameter(1, 3, new EuclidianDistance()),
                    bupa));
            Assert.AreEqual(34.76, 100 * experimentPerformance.MeanPerformance().GetErrorRate(), 0.01);
            experimentPerformance = stratifiedKFoldRun.Execute(new Classification.Experiment.Experiment(new KnnModel(),
                new KnnParameter(1, 3, new EuclidianDistance()), dermatology));
            Assert.AreEqual(10.42, 100 * experimentPerformance.MeanPerformance().GetErrorRate(), 0.01);
            experimentPerformance = stratifiedKFoldRun.Execute(new Classification.Experiment.Experiment(new LdaModel(), new Parameter(1), bupa));
            Assert.AreEqual(31.84, 100 * experimentPerformance.MeanPerformance().GetErrorRate(), 0.01);
            experimentPerformance =
                stratifiedKFoldRun.Execute(new Classification.Experiment.Experiment(new LdaModel(), new Parameter(1), dermatology));
            Assert.AreEqual(2.73, 100 * experimentPerformance.MeanPerformance().GetErrorRate(), 0.01);
            experimentPerformance = stratifiedKFoldRun.Execute(new Classification.Experiment.Experiment(new LinearPerceptronModel(),
                new LinearPerceptronParameter(1, 0.1, 0.99, 0.2, 100), iris));
            Assert.AreEqual(4.67, 100 * experimentPerformance.MeanPerformance().GetErrorRate(), 0.01);
            experimentPerformance = stratifiedKFoldRun.Execute(new Classification.Experiment.Experiment(new LinearPerceptronModel(),
                new LinearPerceptronParameter(1, 0.1, 0.99, 0.2, 100), dermatology));
            Assert.AreEqual(8.90, 100 * experimentPerformance.MeanPerformance().GetErrorRate(), 0.01);
            experimentPerformance = stratifiedKFoldRun.Execute(new Classification.Experiment.Experiment(new NaiveBayesModel(), new Parameter(1), car));
            Assert.AreEqual(14.00, 100 * experimentPerformance.MeanPerformance().GetErrorRate(), 0.01);
            experimentPerformance =
                stratifiedKFoldRun.Execute(new Classification.Experiment.Experiment(new NaiveBayesModel(), new Parameter(1), nursery));
            Assert.AreEqual(9.75, 100 * experimentPerformance.MeanPerformance().GetErrorRate(), 0.01);
            experimentPerformance =
                stratifiedKFoldRun.Execute(new Classification.Experiment.Experiment(new BaggingModel(), new BaggingParameter(1, 50), tictactoe));
            Assert.AreEqual(4.48, 100 * experimentPerformance.MeanPerformance().GetErrorRate(), 0.01);
            experimentPerformance =
                stratifiedKFoldRun.Execute(new Classification.Experiment.Experiment(new BaggingModel(), new BaggingParameter(1, 50), car));
            Assert.AreEqual(6.19, 100 * experimentPerformance.MeanPerformance().GetErrorRate(), 0.01);
            experimentPerformance = stratifiedKFoldRun.Execute(new Classification.Experiment.Experiment(new DummyModel(), new Parameter(1), nursery));
            Assert.AreEqual(66.67, 100 * experimentPerformance.MeanPerformance().GetErrorRate(), 0.01);
            experimentPerformance = stratifiedKFoldRun.Execute(new Classification.Experiment.Experiment(new DummyModel(), new Parameter(1), iris));
            Assert.AreEqual(66.67, 100 * experimentPerformance.MeanPerformance().GetErrorRate(), 0.01);
        }
    }
}