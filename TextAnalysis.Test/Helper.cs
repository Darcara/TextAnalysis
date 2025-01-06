namespace TextAnalysis.Test;

using System.Net.Http;
using System.Runtime.InteropServices;
using System.Threading.Tasks;

public static class Helper {
	public static async Task DownloadTestData() {
		(String uri, String target)[] requiredFiles = [
			("https://huggingface.co/onnx-community/opus-mt-en-de/resolve/main/source.spm?download=true", "data/en-de/source.spm"),
			("https://huggingface.co/onnx-community/opus-mt-en-de/resolve/main/vocab.json?download=true", "data/en-de/vocab.json"),
			("https://huggingface.co/onnx-community/opus-mt-en-de/resolve/main/onnx/decoder_model.onnx?download=true", "data/en-de/decoder_model.onnx"),
			("https://huggingface.co/onnx-community/opus-mt-en-de/resolve/main/onnx/decoder_with_past_model.onnx?download=true", "data/en-de/decoder_with_past_model.onnx"),
			("https://huggingface.co/onnx-community/opus-mt-en-de/resolve/main/onnx/encoder_model.onnx?download=true", "data/en-de/encoder_model.onnx"),
			("https://huggingface.co/segment-any-text/sat-1l-sm/resolve/main/model.onnx?download=true", "data/sat1lsm.onnx"),
			("https://huggingface.co/segment-any-text/sat-3l-sm/resolve/main/model.onnx?download=true", "data/sat3lsm.onnx"),
			("https://huggingface.co/segment-any-text/sat-12l-sm/resolve/main/model.onnx?download=true", "data/sat12lsm.onnx"),
			("https://huggingface.co/FacebookAI/xlm-roberta-base/resolve/main/sentencepiece.bpe.model?download=true", "data/xlm-roberta-base-sentencepiece.bpe.model"),
		];

		using HttpClient client = new();

		foreach ((String uri, String target) requiredFile in requiredFiles) {
			Uri uri = new(requiredFile.uri);
			if (File.Exists(requiredFile.target)) continue;

			Console.WriteLine($"Downloading {requiredFile.target} from {uri}");
			String targetFileAbs = Path.GetFullPath(requiredFile.target);
			Directory.CreateDirectory(Path.GetDirectoryName(targetFileAbs) ?? ".");
			String tempFile = targetFileAbs + ".tmp";
			await using (Stream netStream = await client.GetStreamAsync(uri).ConfigureAwait(false)) {
				await using FileStream fileStream = File.Open(tempFile, FileMode.Create, FileAccess.Write, FileShare.None);
				await netStream.CopyToAsync(fileStream).ConfigureAwait(false);
			}

			File.Move(tempFile, targetFileAbs, true);
		}
	}
	
	internal static void EnsureNativeFilesPresent() {
		switch (RuntimeInformation.RuntimeIdentifier) {
			case "win-x64": 
				if(!File.Exists("./SentencePieceWrapper.dll")) File.Copy("../../../../../MarianTokenizer/runtimes/win-x64/native/SentencePieceWrapper.dll", "./SentencePieceWrapper.dll");
				if(!File.Exists("./sentencepiece.lib")) File.Copy("../../../../../MarianTokenizer/runtimes/win-x64/native/sentencepiece.lib", "./sentencepiece.lib");
				break;
			case "linux-x64": 
				if(!File.Exists("./SentencePieceWrapper.so")) File.Copy("../../../../../MarianTokenizer/runtimes/win-x64/native/SentencePieceWrapper.so", "./SentencePieceWrapper.so");
				if(!File.Exists("./sentencepiece.so")) File.Copy("../../../../../MarianTokenizer/runtimes/win-x64/native/sentencepiece.so", "./sentencepiece.so");
				break;
			default: throw new InvalidOperationException($"Unsupported runtime id: {RuntimeInformation.RuntimeIdentifier}");
		}
	}
}