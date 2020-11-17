using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.ML;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using WEB_APPS.Shared;

namespace WEB_APPS.Server.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class ImageClassificationController : ControllerBase
    {
        private readonly PredictionEnginePool<InMemoryImageData, ImagePrediction> _predictionEnginePool;

        private readonly ILogger<ImageClassificationController> _logger;

        private readonly IHostEnvironment _environment;

        public ImageClassificationController(PredictionEnginePool<InMemoryImageData, ImagePrediction> predictionEnginePool,
                                             IHostEnvironment environment,
                                             ILogger<ImageClassificationController> logger)
        {
            _predictionEnginePool = predictionEnginePool;
            _logger = logger;
            _environment = environment;
        }

        [HttpPost]
        [ProducesResponseType(200)]
        [ProducesResponseType(400)]
        [Route("classifyimage")]
        public async Task<IActionResult> ClassifyImage([FromForm] IFormFile Image)
        {
            if (Image.Length == 0)
                return BadRequest();

            MemoryStream imageMemoryStream = new MemoryStream();

            await Image.CopyToAsync(imageMemoryStream);

            byte[] imageData = imageMemoryStream.ToArray();

            if (!imageData.IsValidImage())
                return StatusCode(StatusCodes.Status415UnsupportedMediaType);

            _logger.LogInformation("Start Processing Image ....");

            Stopwatch watch = Stopwatch.StartNew();

            InMemoryImageData imageInputData = new InMemoryImageData(image: imageData, label: null, imageFileName: null);

            var prediction = _predictionEnginePool.Predict(imageInputData);

            watch.Stop();

            var elapsedMs = watch.ElapsedMilliseconds;

            _logger.LogInformation($"Image processed in { elapsedMs } miliseconds");

            string filePath = Path.Combine(_environment.ContentRootPath, "wwwroot", "Images", Image.FileName);

            using (var fileStream = new FileStream(filePath, FileMode.Create, FileAccess.Write))
            {
                await Image.CopyToAsync(fileStream);
            }

            ImagePredictedLabelWithProbability imageBestLabelPrediction = new ImagePredictedLabelWithProbability
            {
                PredictedLabel = prediction.PredictedLabel,
                Probability = prediction.Score.Max(),
                PredictionExecutionTime = elapsedMs,
                ImageId = Image.FileName
            };

            return Ok(imageBestLabelPrediction);
        }
    }
}