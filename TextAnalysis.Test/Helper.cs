namespace TextAnalysis.Test;

using System.Net.Http;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using Neco.Common.Extensions;

public static class Helper {
	public static async Task DownloadTestData() {
		return;
		(String uri, String target)[] requiredFiles = [
			("https://huggingface.co/onnx-community/opus-mt-en-de/resolve/main/source.spm?download=true", "data/en-de/source.spm"),
			("https://huggingface.co/onnx-community/opus-mt-en-de/resolve/main/vocab.json?download=true", "data/en-de/vocab.json"),
			("https://huggingface.co/onnx-community/opus-mt-en-de/resolve/main/onnx/decoder_model.onnx?download=true", "data/en-de/decoder_model.onnx"),
			("https://huggingface.co/onnx-community/opus-mt-en-de/resolve/main/onnx/decoder_with_past_model.onnx?download=true", "data/en-de/decoder_with_past_model.onnx"),
			("https://huggingface.co/onnx-community/opus-mt-en-de/resolve/main/onnx/encoder_model.onnx?download=true", "data/en-de/encoder_model.onnx"),
			
			("https://huggingface.co/onnx-community/opus-mt-de-en/resolve/main/source.spm?download=true", "data/de-en/source.spm"),
			("https://huggingface.co/onnx-community/opus-mt-de-en/resolve/main/vocab.json?download=true", "data/de-en/vocab.json"),
			("https://huggingface.co/onnx-community/opus-mt-de-en/resolve/main/onnx/decoder_model.onnx?download=true", "data/de-en/decoder_model.onnx"),
			("https://huggingface.co/onnx-community/opus-mt-de-en/resolve/main/onnx/decoder_with_past_model.onnx?download=true", "data/de-en/decoder_with_past_model.onnx"),
			("https://huggingface.co/onnx-community/opus-mt-de-en/resolve/main/onnx/encoder_model.onnx?download=true", "data/de-en/encoder_model.onnx"),
			
			// ("https://huggingface.co/segment-any-text/sat-1l-sm/resolve/main/model.onnx?download=true", "data/sat1lsm.onnx"),
			// ("https://huggingface.co/segment-any-text/sat-3l-sm/resolve/main/model.onnx?download=true", "data/sat3lsm.onnx"),
			// ("https://huggingface.co/segment-any-text/sat-12l-sm/resolve/main/model.onnx?download=true", "data/sat12lsm.onnx"),
			// ("https://huggingface.co/FacebookAI/xlm-roberta-base/resolve/main/sentencepiece.bpe.model?download=true", "data/xlm-roberta-base-sentencepiece.bpe.model"),
			
			// ("https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin", "data/lid.176.bin"),
			// ("https://dl.fbaipublicfiles.com/nllb/lid/lid218e.bin", "data/lid.218e.bin"),
			// Alt download for the same model
			// ("https://huggingface.co/facebook/fasttext-language-identification/resolve/main/model.bin?download=true", "data/lid.lid218e.bin"),
		];

		using HttpClient client = new();

		foreach ((String uri, String target) requiredFile in requiredFiles) {
			await DownloadFile(client, requiredFile.uri,requiredFile.target, fi => !fi.Exists);
		}
	}

	internal static Lazy<String> WebFile(String file, String source) {
		FileInfo fi = new(file);
		if (fi.Exists) 
			return new(file);
		
		return new(() => DownloadFile(source, file).GetResultBlocking(), LazyThreadSafetyMode.ExecutionAndPublication);
	}

	internal static Task<String> DownloadFile(String uri, String destination, Predicate<FileInfo> ? destinationPredicate = null) => DownloadFile(new Uri(uri), destination, destinationPredicate);

	internal static async Task<String> DownloadFile(Uri uri, String destination, Predicate<FileInfo> ? destinationPredicate = null) {
		using HttpClient client = new();
		return await DownloadFile(client, uri, destination, destinationPredicate);
	}

	internal static Task<String> DownloadFile(HttpClient client, String uri, String destination, Predicate<FileInfo> ? destinationPredicate = null) => DownloadFile(client, new Uri(uri), destination, destinationPredicate);

	internal static async Task<String> DownloadFile(HttpClient client, Uri uri, String destination, Predicate<FileInfo>? destinationPredicate = null) {
		FileInfo fi = new(destination);
		destinationPredicate ??= fileInfo => !fileInfo.Exists;
		if (!destinationPredicate(fi)) return destination;

		Console.WriteLine($"Downloading {destination} from {uri}");
		String targetFileAbs = Path.GetFullPath(destination);
		Directory.CreateDirectory(Path.GetDirectoryName(targetFileAbs) ?? ".");
		String tempFile = targetFileAbs + ".tmp";
		await using (Stream netStream = await client.GetStreamAsync(uri).ConfigureAwait(false)) {
			await using FileStream fileStream = File.Open(tempFile, FileMode.Create, FileAccess.Write, FileShare.None);
			await netStream.CopyToAsync(fileStream).ConfigureAwait(false);
		}

		File.Move(tempFile, targetFileAbs, true);
		return destination;
	}

	internal static void EnsureNativeFilesPresent() {
		return;
		switch (RuntimeInformation.RuntimeIdentifier) {
			case "win-x64":
				if (!File.Exists("./SentencePieceWrapper.dll")) File.Copy("../../../../../MarianTokenizer/runtimes/win-x64/native/SentencePieceWrapper.dll", "./SentencePieceWrapper.dll");
				if (!File.Exists("./sentencepiece.lib")) File.Copy("../../../../../MarianTokenizer/runtimes/win-x64/native/sentencepiece.lib", "./sentencepiece.lib");
				break;
			case "linux-x64":
				if (!File.Exists("./SentencePieceWrapper.so")) File.Copy("../../../../../MarianTokenizer/runtimes/win-x64/native/SentencePieceWrapper.so", "./SentencePieceWrapper.so");
				if (!File.Exists("./sentencepiece.so")) File.Copy("../../../../../MarianTokenizer/runtimes/win-x64/native/sentencepiece.so", "./sentencepiece.so");
				break;
			default: throw new InvalidOperationException($"Unsupported runtime id: {RuntimeInformation.RuntimeIdentifier}");
		}
	}
}