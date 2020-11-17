namespace WEB_APPS.Shared
{
    public class ImagePredictedLabelWithProbability
    {
        public string ImageId { get; set; }

        public string PredictedLabel { get; set; }

        public float Probability { get; set; }

        public long PredictionExecutionTime { get; set; }
    }
}