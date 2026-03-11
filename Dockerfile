FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build
WORKDIR /src
#COPY *.sln .
#COPY ["TextAnalysis/TextAnalysis.csproj", "TextAnalysis/"]
#COPY ["TextAnalysis.Benchmark/TextAnalysis.Benchmark.csproj", "TextAnalysis.Benchmark/"]
#COPY ["WebTranslator/WebTranslator.csproj", "WebTranslator/"]
#COPY ["TextAnalysis.Test/TextAnalysis.Test.csproj", "TextAnalysis.Test/"]

ENV DOTNET_NUGET_SIGNATURE_VERIFICATION=false

COPY . .

RUN du -sh .
RUN du -sh /root/.nuget/

RUN --mount=type=cache,id=nugetpackages,target=/root/.nuget/packages \
    dotnet restore --nologo --disable-build-servers --runtime linux-x64
RUN #dotnet restore --nologo --disable-build-servers --runtime linux-x64

RUN du -sh .
RUN du -sh /root/.nuget/

#COPY . .
RUN --mount=type=cache,id=nugetpackages,target=/root/.nuget/packages \
    dotnet build TextAnalysis.Test --no-restore --nologo --disable-build-servers --runtime linux-x64

#RUN --mount=type=cache,id=nugetpackages,target=/root/.nuget/packages \
#    --mount=type=cache,id=textanalysisdata,target=/src/TextAnalysis.Test/bin/Debug/net10.0/linux-x64/data \
#    dotnet test TextAnalysis.Test --nologo --no-restore --no-build --runtime linux-x64 --filter:Name~SatSplitterExample.WikiText

#FROM build AS testrunner
#WORKDIR /src/

#CMD ["dotnet", "test", "--nologo", "--no-restore", "--no-build", "--tl:off", "--filter:Category!~Benchmark"]
CMD ["dotnet", "test", "TextAnalysis.Test", "--nologo", "--no-restore", "--no-build", "--runtime:linux-x64", "--filter:Name~SatSplitterExample.WikiText"]
#dotnet test --no-build --filter:Name~SatSplitterExample.WikiText
#CMD ["echo", "done"]
#CMD ["bash"]


