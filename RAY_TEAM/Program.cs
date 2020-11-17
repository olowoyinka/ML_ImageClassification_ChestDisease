using Microsoft.ML;
using Microsoft.ML.Vision;
using RAY_TEAM.Helper;
using RAY_TEAM.Models;
using System;
using System.Collections.Generic;
using System.IO;
using static Microsoft.ML.Transforms.ValueToKeyMappingEstimator;

namespace RAY_TEAM
{
    class Program
    {
        private static string _appPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);

        private static string projectDirectory = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../"));
        
        private static string _trainDataPath = Path.Combine(projectDirectory, "trainingdata");
        
        private static string _validationDataPath = Path.Combine(projectDirectory, "validationdata");

        static string _modelPath => Path.Combine(_appPath, "..", "..", "..", "Models", "TrainedModel.zip");

        static void Main(string[] args)
        {
            MLContext _mLContext = new MLContext();

            IEnumerable<ImageData> trainingimages = LoadImagesFromDirectory(folder: _trainDataPath, useFolderNameAslabel: true);

            IEnumerable<ImageData> validationimages = LoadImagesFromDirectory(folder: _validationDataPath, useFolderNameAslabel: true);

            IDataView trainingimageData = _mLContext.Data.LoadFromEnumerable(trainingimages);

            IDataView validationimageData = _mLContext.Data.LoadFromEnumerable(validationimages);

            IDataView shuffledtrainingImageData = _mLContext.Data.ShuffleRows(trainingimageData);
            
            IDataView shuffledvalidationImageData = _mLContext.Data.ShuffleRows(validationimageData);

            IDataView trainDataView = _mLContext.Transforms.Conversion.MapValueToKey(outputColumnName: "LabelAsKey", inputColumnName: "Label", keyOrdinality: KeyOrdinality.ByValue)
                                                            .Append(_mLContext.Transforms.LoadRawImageBytes(outputColumnName: "Image", imageFolder: _trainDataPath, inputColumnName: "ImagePath"))
                                                            .Fit(shuffledtrainingImageData)
                                                            .Transform(shuffledtrainingImageData);

            IDataView testDataView = _mLContext.Transforms.Conversion.MapValueToKey(outputColumnName: "LabelAsKey", inputColumnName: "Label", keyOrdinality: KeyOrdinality.ByValue)
                                                            .Append(_mLContext.Transforms.LoadRawImageBytes(outputColumnName: "Image", imageFolder: _validationDataPath, inputColumnName: "ImagePath"))
                                                            .Fit(shuffledvalidationImageData)
                                                            .Transform(shuffledvalidationImageData);

            var options = new ImageClassificationTrainer.Options()
            {
                FeatureColumnName = "Image",
                LabelColumnName = "LabelAsKey",
                Arch = ImageClassificationTrainer.Architecture.ResnetV2101,
                Epoch = 50,
                BatchSize = 10,
                LearningRate = 0.01f,
                MetricsCallback = (metrics) => Console.WriteLine(metrics),
                ValidationSet = testDataView
            };

            IEstimator<ITransformer> trainingpipeLine = _mLContext.MulticlassClassification.Trainers.ImageClassification(options)
                                                           .Append(_mLContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            ITransformer trainedModel = trainingpipeLine.Fit(trainDataView);

            EvaluateModel(_mLContext, testDataView, trainedModel);

            _mLContext.Model.Save(trainedModel, trainDataView.Schema, _modelPath);
        }

        private static void EvaluateModel(MLContext mlContext, IDataView testDataset, ITransformer trainedModel)
        {
            Console.WriteLine("Making predictions in bulk for evaluating model's quality...");

            var predictionsDataView = trainedModel.Transform(testDataset);

            var metrics = mlContext.MulticlassClassification.Evaluate(predictionsDataView, labelColumnName: "LabelAsKey", predictedLabelColumnName: "PredictedLabel");

            ConsoleHelper.PrintMultiClassClassificationMetrics("TensorFlow DNN Transfer Learning", metrics);
        }

        public static IEnumerable<ImageData> LoadImagesFromDirectory(string folder, bool useFolderNameAslabel = true)
        {
            var files = Directory.GetFiles(folder, "*", searchOption: SearchOption.AllDirectories);

            foreach (var file in files)
            {
                if ((Path.GetExtension(file) != ".jpg") && (Path.GetExtension(file) != ".png"))
                    continue;

                var label = Path.GetFileName(file);

                if (useFolderNameAslabel)
                {
                    label = Directory.GetParent(file).Name;
                }
                else
                {
                    for (int index = 0; index < label.Length; index++)
                    {
                        if (!char.IsLetter(label[index]))
                        {
                            label = label.Substring(0, index);

                            break;
                        }
                    }
                }

                yield return new ImageData()
                {
                    ImagePath = file,
                    Label = label
                };
            }
        }
    }
}