// significant portions of this file were derived from this project
/*****************************************************************
|
|    AP4 - MP4 File Info
|
|    Copyright 2002-2015 Axiomatic Systems, LLC
|
|
|    This file is part of Bento4/AP4 (MP4 Atom Processing Library).
|
|    Unless you have obtained Bento4 under a difference license,
|    this version of Bento4 is Bento4|GPL.
|    Bento4|GPL is free software; you can redistribute it and/or modify
|    it under the terms of the GNU General Public License as published by
|    the Free Software Foundation; either version 2, or (at your option)
|    any later version.
|
|    Bento4|GPL is distributed in the hope that it will be useful,
|    but WITHOUT ANY WARRANTY; without even the implied warranty of
|    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
|    GNU General Public License for more details.
|
|    You should have received a copy of the GNU General Public License
|    along with Bento4|GPL; see the file COPYING.  If not, write to the
|    Free Software Foundation, 59 Temple Place - Suite 330, Boston, MA
|    02111-1307, USA.
|
 ****************************************************************/
#include "ExampleXGL.h"
#include "xbento4_class.h"

void XBento4::Draw()
{
	glProgramUniform1i(shader->programId, glGetUniformLocation(shader->programId, "texUnit0"), 0);
	glProgramUniform1i(shader->programId, glGetUniformLocation(shader->programId, "texUnit1"), 1);
	glProgramUniform1i(shader->programId, glGetUniformLocation(shader->programId, "texUnit2"), 2);

	XGLTexQuad::Draw();
}


XBento4::XBento4() : XThread("XBento4Thread")
{
	SetName("XBento4");

}

XBento4::XBento4(std::string fname) : filename(fname), XThread("XBento4Thread"), XGLTexQuad()
{
	//XBento4();
	SetName("XBento4");

	AP4_Result result = AP4_FileByteStream::Create(
		filename.c_str(),
		AP4_FileByteStream::STREAM_MODE_READ,
		input);

	if (AP4_FAILED(result))
		throw std::runtime_error("Oops: AP4_FileBytestreamCreate() didn't");

	AP4_File* file = new AP4_File(*input, true);
	ShowFileInfo(*file);

	AP4_Movie* movie = file->GetMovie();
	if (movie)
		ShowMovieInfo(*movie);

	AP4_List<AP4_Track>& tracks = movie->GetTracks();

	MediaInfo mediaInfo;
	MakeTrackList(*movie, tracks, *input, mediaInfo);

	//Stop();  // clear "isRunning" in our XThread
	xprintf("%s() done.\n", __FUNCTION__);

	FILE *yuvInputFile = fopen("test_dec.yuv", "rb");
	if (yuvInputFile) {
		int nRead = fread(yuvBuffer, 1, sizeof(yuvBuffer), yuvInputFile);
		if (nRead != sizeof(yuvBuffer))
		{
			xprintf("Dang, didn't init the yuvBuffer\n");
		}
		else
		{
			unsigned char *y = yuvBuffer;
			unsigned char *u = yuvBuffer + 1920 * 1080;
			unsigned char *v = u + 1920 * 1080 / 4;

			AddTexture(1920, 1080, 1, y);
			AddTexture(960, 540, 1, u);
			AddTexture(960, 540, 1, v);
		}

		fclose(yuvInputFile);
	}
}

XBento4::~XBento4()
{
	if(input)
		delete input;
}

void XBento4::MakeTrackList(AP4_Movie& movie, AP4_List<AP4_Track>&in, AP4_ByteStream& stream, MediaInfo&)
{
	for (auto t = in.FirstItem(); t; t = t->GetNext())
		trackList.push_back(t->GetData());

	xprintf("%s(): loaded up %d tracks\n", __FUNCTION__, trackList.size());

	auto pt = trackList[0];
	{
		AP4_Track& t{ *pt };
		xprintf("Track type: %s, %d samples\n",
			trackTypes[t.GetType()],
			t.GetSampleCount());

		AP4_SampleDescription* sample_description = t.GetSampleDescription(0);
		if (sample_description->GetType() == AP4_SampleDescription::TYPE_AVC)
			xprintf("Yippie: it's an AVC sample!\n");

		Samples* samples = new Samples();
		AP4_Sample sample;
		AP4_DataBuffer sample_data;

		for (int idx = 0; idx < t.GetSampleCount(); idx++)
		{
			t.GetSample(idx, sample);
			samples->push_back(sample);
			//sample.ReadData(sample_data);
			//sample_data.GetDataSize();
		}
		sl.push_back(samples);
	}

	if (false)
	{
		xprintf("sl.size(): %d\n", sl.size());

		XGLColor colors[4]
		{
			XGLColors::green,
			XGLColors::yellow,
			XGLColors::cyan,
			XGLColors::magenta
		};

		int i{ 0 };
		for (Samples* samples : sl)
		{
			xprintf("Track %d: %d samples\n", i++, samples->size());
			for (AP4_Sample sample : *samples)
			{
				float y = 1.0f * i;
				float x = (float)sample.GetOffset() / 20000000.0f;
				float z = (float)sample.GetSize() / 20000.0f;


				v.push_back({ {x, y, 0}, {}, {}, colors[i] });
				v.push_back({ {x, y, z}, {}, {}, colors[i] });
			}
		}
	}
	return;
}

void XBento4::Run() {
	AP4_Result result = AP4_FileByteStream::Create(
		filename.c_str(),
		AP4_FileByteStream::STREAM_MODE_READ,
		input);

	if (AP4_FAILED(result))
		throw std::runtime_error("Oops: AP4_FileBytestreamCreate() didn't");

	AP4_File* file = new AP4_File(*input, true);
	ShowFileInfo(*file);

	AP4_Movie* movie = file->GetMovie();
	if (movie)
		ShowMovieInfo(*movie);

	AP4_List<AP4_Track>& tracks = movie->GetTracks();

	MediaInfo mediaInfo;
	MakeTrackList(*movie, tracks, *input, mediaInfo);

	//Stop();  // clear "isRunning" in our XThread
	xprintf("%s() done.\n", __FUNCTION__);
}

void XBento4::ShowFileInfo(AP4_File& file)
{
	AP4_FtypAtom* file_type{ nullptr };
	
	if ( (file_type = file.GetFileType()) == NULL) 
		return;

	char four_cc[5];

	AP4_FormatFourChars(four_cc, file_type->GetMajorBrand());
	xprintf("File:\n");
	xprintf("  major brand:      %s\n", four_cc);
	xprintf("  minor version:    %x\n", file_type->GetMinorVersion());
}

void XBento4::ShowMovieInfo(AP4_Movie& movie)
{
	xprintf("Movie:\n");
	xprintf("  duration:   %d ms\n", movie.GetDurationMs());
	xprintf("  time scale: %d\n", movie.GetTimeScale());
	xprintf("  fragments:  %s\n", movie.HasFragments() ? "yes" : "no");
}

void XBento4::ShowTracks
(
	AP4_Movie& movie, 
	AP4_List<AP4_Track>& tracks, 
	AP4_ByteStream& stream, 
	bool show_samples, 
	bool show_sample_data, 
	bool verbose, 
	bool fast
)
{
	int index = 1;
	for (AP4_List<AP4_Track>::Item* track_item = tracks.FirstItem();
		track_item;
		track_item = track_item->GetNext(), ++index) 
	{
		xprintf("Track %d:\n", index);
		ShowTrackInfo(movie, *track_item->GetData(), stream, show_samples, show_sample_data, verbose, fast);
	}
}

void XBento4::ShowTrackInfo(
	AP4_Movie& movie,
	AP4_Track& track,
	AP4_ByteStream& stream,
	bool show_samples,
	bool show_sample_data,
	bool verbose,
	bool fast)
{
	xprintf("  flags:        %d", track.GetFlags());
	if (track.GetFlags() & AP4_TRACK_FLAG_ENABLED)
	{
		xprintf(" ENABLED");
	}
	if (track.GetFlags() & AP4_TRACK_FLAG_IN_MOVIE)
	{
		xprintf(" IN-MOVIE");
	}
	if (track.GetFlags() & AP4_TRACK_FLAG_IN_PREVIEW)
	{
		xprintf(" IN-PREVIEW");
	}
	xprintf("\n");
	xprintf("  id:           %d\n", track.GetId());
	xprintf("  type:         ");

	switch (track.GetType()) {
		case AP4_Track::TYPE_AUDIO:     xprintf("Audio\n");     break;
		case AP4_Track::TYPE_VIDEO:     xprintf("Video\n");     break;
		case AP4_Track::TYPE_HINT:      xprintf("Hint\n");      break;
		case AP4_Track::TYPE_SYSTEM:    xprintf("System\n");    break;
		case AP4_Track::TYPE_TEXT:      xprintf("Text\n");      break;
		case AP4_Track::TYPE_JPEG:      xprintf("JPEG\n");      break;
		case AP4_Track::TYPE_SUBTITLES: xprintf("Subtitles\n"); break;
		default: {
			char hdlr[5];
			AP4_FormatFourChars(hdlr, track.GetHandlerType());
			xprintf("Unknown [");
			xprintf("%s", hdlr);
			xprintf("]\n");
			break;
		}
	}

	xprintf("  duration: %d ms\n", track.GetDurationMs());
	xprintf("  language: %s\n", track.GetTrackLanguage());
	xprintf("  media:\n");
	xprintf("    sample count: %d\n", track.GetSampleCount());
	xprintf("    timescale:    %d\n", track.GetMediaTimeScale());
	xprintf("    duration:     %lld (media timescale units)\n", track.GetMediaDuration());
	xprintf("    duration:     %d (ms)\n", (AP4_UI32)AP4_ConvertTime(track.GetMediaDuration(), track.GetMediaTimeScale(), 1000));

	if (!fast)
	{
		MediaInfo media_info;
		ScanMedia(movie, track, stream, media_info);
		xprintf("    bitrate (computed): %.3f Kbps\n", media_info.bitrate / 1000.0);
		if (movie.HasFragments())
		{
			xprintf("    sample count with fragments: %lld\n", media_info.sample_count);
			xprintf("    duration with fragments:     %lld\n", media_info.duration);
			xprintf("    duration with fragments:     %d (ms)\n", (AP4_UI32)AP4_ConvertTime(media_info.duration, track.GetMediaTimeScale(), 1000));
		}
	}
	if (track.GetWidth() || track.GetHeight())
	{
		xprintf("  display width:  %f\n", (float)track.GetWidth() / 65536.0);
		xprintf("  display height: %f\n", (float)track.GetHeight() / 65536.0);
	}
	if (track.GetType() == AP4_Track::TYPE_VIDEO && track.GetSampleCount())
	{
		xprintf("  frame rate (computed): %.3f\n", (float)track.GetSampleCount() /
			((float)track.GetMediaDuration() / (float)track.GetMediaTimeScale()));
	}
	// show all sample descriptions
	AP4_AvcSampleDescription* avc_desc = NULL;
	for (unsigned int desc_index = 0;
		AP4_SampleDescription* sample_desc = track.GetSampleDescription(desc_index);
		desc_index++)
	{
		xprintf("  Sample Description %d\n", desc_index);
		ShowSampleDescription(*sample_desc, verbose);
		avc_desc = AP4_DYNAMIC_CAST(AP4_AvcSampleDescription, sample_desc);
	}

	// show samples if requested
	if (show_samples)
	{
		AP4_Sample     sample;
		AP4_DataBuffer sample_data;
		AP4_Ordinal    index = 0;
		while (AP4_SUCCEEDED(track.GetSample(index, sample)))
		{
			if (avc_desc || show_sample_data)
			{
				sample.ReadData(sample_data);
			}

			ShowSample(track, sample, sample_data, index, verbose, show_sample_data, avc_desc);
			xprintf("\n");
			index++;
		}
	}
}

void XBento4::ShowSample(
	AP4_Track&      track,
	AP4_Sample&     sample,
	AP4_DataBuffer& sample_data,
	unsigned int    index,
	bool            verbose,
	bool            show_sample_data,
	AP4_AvcSampleDescription* avc_desc)
{
	xprintf("[%06d] size=%6d duration=%6d",
		index + 1,
		(int)sample.GetSize(),
		(int)sample.GetDuration());
	if (verbose)
	{
		xprintf(" (%6d ms) offset=%10lld dts=%10lld (%10lld ms) cts=%10lld (%10lld ms) [%d]",
			(int)AP4_ConvertTime(sample.GetDuration(), track.GetMediaTimeScale(), 1000),
			sample.GetOffset(),
			sample.GetDts(),
			AP4_ConvertTime(sample.GetDts(), track.GetMediaTimeScale(), 1000),
			sample.GetCts(),
			AP4_ConvertTime(sample.GetCts(), track.GetMediaTimeScale(), 1000),
			sample.GetDescriptionIndex());
	}
	if (sample.IsSync())
	{
		xprintf(" [S] ");
	}
	else
	{
		xprintf("     ");
	}
	if (avc_desc || show_sample_data)
	{
		sample.ReadData(sample_data);
	}
	if (avc_desc)
	{
		ShowAvcInfo(sample_data, avc_desc);
	}
	if (show_sample_data)
	{
		unsigned int show = sample_data.GetDataSize();
		if (!verbose)
		{
			if (show > 12) show = 12; // max first 12 chars
		}

		for (unsigned int i = 0; i < show; i++)
		{
			if (verbose)
			{
				if (i % 16 == 0)
				{
					xprintf("\n%06d: ", i);
				}
			}
			xprintf("%02x", sample_data.GetData()[i]);
			if (verbose)
				xprintf(" ");
		}

		if (show != sample_data.GetDataSize())
		{
			xprintf("...");
		}
	}
}

void XBento4::ShowAvcInfo(const AP4_DataBuffer& sample_data, AP4_AvcSampleDescription* avc_desc)
{
	const unsigned char* data = sample_data.GetData();
	AP4_Size             size = sample_data.GetDataSize();

	while (size >= avc_desc->GetNaluLengthSize())
	{
		unsigned int nalu_length = 0;
		if (avc_desc->GetNaluLengthSize() == 1)
		{
			nalu_length = *data++;
			--size;
		}
		else if (avc_desc->GetNaluLengthSize() == 2)
		{
			nalu_length = AP4_BytesToUInt16BE(data);
			data += 2;
			size -= 2;
		}
		else if (avc_desc->GetNaluLengthSize() == 4)
		{
			nalu_length = AP4_BytesToUInt32BE(data);
			data += 4;
			size -= 4;
		}
		else
		{
			return;
		}
		if (nalu_length <= size)
		{
			size -= nalu_length;
		}
		else 
		{
			size = 0;
		}

		switch (*data & 0x1F) {
			case 1: {
				AP4_BitStream bits;
				bits.WriteBytes(data + 1, 8);
				ReadGolomb(bits);
				unsigned int slice_type = ReadGolomb(bits);
				switch (slice_type) {
					case 0: xprintf("<P>");  break;
					case 1: xprintf("<B>");  break;
					case 2: xprintf("<I>");  break;
					case 3:	xprintf("<SP>"); break;
					case 4: xprintf("<SI>"); break;
					case 5: xprintf("<P>");  break;
					case 6: xprintf("<B>");  break;
					case 7: xprintf("<I>");  break;
					case 8:	xprintf("<SP>"); break;
					case 9: xprintf("<SI>"); break;
					default: xprintf("<S/%d>", slice_type); break;
				}
				return; // only show first slice type
			}

			case 5:
				xprintf("<I>");
				return;
		}

		data += nalu_length;
	}
}

unsigned int XBento4::ReadGolomb(AP4_BitStream& bits)
{
	unsigned int leading_zeros = 0;
	while (bits.ReadBit() == 0)
	{
		leading_zeros++;
	}
	if (leading_zeros)
	{
		return (1 << leading_zeros) - 1 + bits.ReadBits(leading_zeros);
	}
	else
	{
		return 0;
	}
}

void XBento4::ShowSampleDescription(AP4_SampleDescription& description, bool verbose)
{
	AP4_SampleDescription* desc = &description;
	if (desc->GetType() == AP4_SampleDescription::TYPE_PROTECTED)
	{
		AP4_ProtectedSampleDescription* prot_desc = AP4_DYNAMIC_CAST(AP4_ProtectedSampleDescription, desc);
		if (prot_desc)
		{
			ShowProtectedSampleDescription_Text(*prot_desc, verbose);
			desc = prot_desc->GetOriginalSampleDescription();
		}
	}
	if (verbose)
	{
		xprintf("    Bytes: ");
		AP4_Atom* details = desc->ToAtom();
		ShowPayload(*details, false);
		xprintf("\n");
		delete details;
	}

	char coding[5];
	AP4_FormatFourChars(coding, desc->GetFormat());
	xprintf("    Coding:      %s", coding);
	const char* format_name = AP4_GetFormatName(desc->GetFormat());
	if (format_name)
	{
		xprintf(" (%s)\n", format_name);
	}
	else
	{
		xprintf("\n");
	}

	if (desc->GetType() == AP4_SampleDescription::TYPE_MPEG)
	{
		// MPEG sample description
		AP4_MpegSampleDescription* mpeg_desc = AP4_DYNAMIC_CAST(AP4_MpegSampleDescription, desc);

		xprintf("    Stream Type: %s\n", mpeg_desc->GetStreamTypeString(mpeg_desc->GetStreamType()));
		xprintf("    Object Type: %s\n", mpeg_desc->GetObjectTypeString(mpeg_desc->GetObjectTypeId()));
		xprintf("    Max Bitrate: %d\n", mpeg_desc->GetMaxBitrate());
		xprintf("    Avg Bitrate: %d\n", mpeg_desc->GetAvgBitrate());
		xprintf("    Buffer Size: %d\n", mpeg_desc->GetBufferSize());

		if (mpeg_desc->GetObjectTypeId() == AP4_OTI_MPEG4_AUDIO ||
			mpeg_desc->GetObjectTypeId() == AP4_OTI_MPEG2_AAC_AUDIO_LC ||
			mpeg_desc->GetObjectTypeId() == AP4_OTI_MPEG2_AAC_AUDIO_MAIN)
		{
			AP4_MpegAudioSampleDescription* mpeg_audio_desc = AP4_DYNAMIC_CAST(AP4_MpegAudioSampleDescription, mpeg_desc);
			if (mpeg_audio_desc) 
				ShowMpegAudioSampleDescription(*mpeg_audio_desc);
		}
	}

	AP4_AudioSampleDescription* audio_desc =
		AP4_DYNAMIC_CAST(AP4_AudioSampleDescription, desc);
	if (audio_desc)
	{
		// Audio sample description
		xprintf("    Sample Rate: %d\n", audio_desc->GetSampleRate());
		xprintf("    Sample Size: %d\n", audio_desc->GetSampleSize());
		xprintf("    Channels:    %d\n", audio_desc->GetChannelCount());
	}
	AP4_VideoSampleDescription* video_desc =
		AP4_DYNAMIC_CAST(AP4_VideoSampleDescription, desc);
	if (video_desc) 
	{
		// Video sample description
		xprintf("    Width:       %d\n", video_desc->GetWidth());
		xprintf("    Height:      %d\n", video_desc->GetHeight());
		xprintf("    Depth:       %d\n", video_desc->GetDepth());
	}

	// Dolby Digital specifics
	if (desc->GetFormat() == AP4_SAMPLE_FORMAT_AC_3)
	{
		AP4_Dac3Atom* dac3 = AP4_DYNAMIC_CAST(AP4_Dac3Atom, desc->GetDetails().GetChild(AP4_ATOM_TYPE_DAC3));
		if (dac3)
		{
			xprintf("    AC-3 Data Rate: %d\n", dac3->GetDataRate());
			xprintf("    AC-3 Stream:\n");
			xprintf("        fscod       = %d\n", dac3->GetStreamInfo().fscod);
			xprintf("        bsid        = %d\n", dac3->GetStreamInfo().bsid);
			xprintf("        bsmod       = %d\n", dac3->GetStreamInfo().bsmod);
			xprintf("        acmod       = %d\n", dac3->GetStreamInfo().acmod);
			xprintf("        lfeon       = %d\n", dac3->GetStreamInfo().lfeon);
			xprintf("    AC-3 dac3 payload: [");
			ShowData(dac3->GetRawBytes());
			xprintf("]\n");
		}
	}

	// Dolby Digital Plus specifics
	if (desc->GetFormat() == AP4_SAMPLE_FORMAT_EC_3)
	{
		AP4_Dec3Atom* dec3 = AP4_DYNAMIC_CAST(AP4_Dec3Atom, desc->GetDetails().GetChild(AP4_ATOM_TYPE_DEC3));
		if (dec3)
		{
			xprintf("    AC-3 Data Rate: %d\n", dec3->GetDataRate());
			for (unsigned int i = 0; i < dec3->GetSubStreams().ItemCount(); i++) {
				xprintf("    AC-3 Substream %d:\n", i);
				xprintf("        fscod       = %d\n", dec3->GetSubStreams()[i].fscod);
				xprintf("        bsid        = %d\n", dec3->GetSubStreams()[i].bsid);
				xprintf("        bsmod       = %d\n", dec3->GetSubStreams()[i].bsmod);
				xprintf("        acmod       = %d\n", dec3->GetSubStreams()[i].acmod);
				xprintf("        lfeon       = %d\n", dec3->GetSubStreams()[i].lfeon);
				xprintf("        num_dep_sub = %d\n", dec3->GetSubStreams()[i].num_dep_sub);
				xprintf("        chan_loc    = %d\n", dec3->GetSubStreams()[i].chan_loc);
			}
			xprintf("    AC-3 dec3 payload: [");
			ShowData(dec3->GetRawBytes());
			xprintf("]\n");
		}
	}

	// Dolby AC-4 specifics
	if (desc->GetFormat() == AP4_SAMPLE_FORMAT_AC_4) 
	{
		AP4_Dac4Atom* dac4 = AP4_DYNAMIC_CAST(AP4_Dac4Atom, desc->GetDetails().GetChild(AP4_ATOM_TYPE_DAC4));
		if (dac4) 
		{
			xprintf("    Codecs String: ");
			AP4_String codec;
			dac4->GetCodecString(codec);
			xprintf("%s", codec.GetChars());
			xprintf("\n");

			const AP4_Dac4Atom::Ac4Dsi& dsi = dac4->GetDsi();
			if (dsi.ac4_dsi_version == 1) 
			{
				for (unsigned int i = 0; i < dsi.d.v1.n_presentations; i++)
				{
					AP4_Dac4Atom::Ac4Dsi::PresentationV1& presentation = dsi.d.v1.presentations[i];
					if (presentation.presentation_version == 1)
					{
						xprintf("    AC-4 Presentation %d:\n", i);
						xprintf("        presentation_channel_mask_v1 = %x\n",
							presentation.d.v1.presentation_channel_mask_v1);
					}
				}
			}

			xprintf("    AC-4 dac4 payload: [");
			ShowData(dac4->GetRawBytes());
			xprintf("]\n");
		}
	}

	// AVC specifics
	if (desc->GetType() == AP4_SampleDescription::TYPE_AVC)
	{
		// AVC Sample Description
		AP4_AvcSampleDescription* avc_desc = AP4_DYNAMIC_CAST(AP4_AvcSampleDescription, desc);
		const char* profile_name = AP4_AvccAtom::GetProfileName(avc_desc->GetProfile());
		xprintf("    AVC Profile:          %d", avc_desc->GetProfile());
		if (profile_name) 
		{
			xprintf(" (%s)\n", profile_name);
		}
		else 
		{
			xprintf("\n");
		}
		xprintf("    AVC Profile Compat:   %x\n", avc_desc->GetProfileCompatibility());
		xprintf("    AVC Level:            %d\n", avc_desc->GetLevel());
		xprintf("    AVC NALU Length Size: %d\n", avc_desc->GetNaluLengthSize());
		xprintf("    AVC SPS: [");

		const char* sep = "";
		for (unsigned int i = 0; i < avc_desc->GetSequenceParameters().ItemCount(); i++) {
			xprintf("%s", sep);
			ShowData(avc_desc->GetSequenceParameters()[i]);
			sep = ", ";
		}
		xprintf("]\n");
		xprintf("    AVC PPS: [");
		sep = "";
		for (unsigned int i = 0; i < avc_desc->GetPictureParameters().ItemCount(); i++)
		{
			xprintf("%s", sep);
			ShowData(avc_desc->GetPictureParameters()[i]);
			sep = ", ";
		}
		xprintf("]\n");
		xprintf("    Codecs String: ");

		AP4_String codec;
		avc_desc->GetCodecString(codec);
		xprintf("%s", codec.GetChars());
		xprintf("\n");
	}
	else if (desc->GetType() == AP4_SampleDescription::TYPE_HEVC) 
	{
		// HEVC Sample Description
		AP4_HevcSampleDescription* hevc_desc = AP4_DYNAMIC_CAST(AP4_HevcSampleDescription, desc);
		const char* profile_name = AP4_HvccAtom::GetProfileName(hevc_desc->GetGeneralProfileSpace(), hevc_desc->GetGeneralProfile());
		xprintf("    HEVC Profile Space:       %d\n", hevc_desc->GetGeneralProfileSpace());
		xprintf("    HEVC Profile:             %d", hevc_desc->GetGeneralProfile());
		if (profile_name) 
			xprintf(" (%s)", profile_name);
		xprintf("\n");

		xprintf("    HEVC Profile Compat:      %x\n", hevc_desc->GetGeneralProfileCompatibilityFlags());
		xprintf("    HEVC Level:               %d.%d\n", hevc_desc->GetGeneralLevel() / 30, (hevc_desc->GetGeneralLevel() % 30) / 3);
		xprintf("    HEVC Tier:                %d\n", hevc_desc->GetGeneralTierFlag());
		xprintf("    HEVC Chroma Format:       %d", hevc_desc->GetChromaFormat());

		const char* chroma_format_name = AP4_HvccAtom::GetChromaFormatName(hevc_desc->GetChromaFormat());
		if (chroma_format_name) 
			xprintf(" (%s)", chroma_format_name);
		xprintf("\n");

		xprintf("    HEVC Chroma Bit Depth:    %d\n", hevc_desc->GetChromaBitDepth());
		xprintf("    HEVC Luma Bit Depth:      %d\n", hevc_desc->GetLumaBitDepth());
		xprintf("    HEVC Average Frame Rate:  %d\n", hevc_desc->GetAverageFrameRate());
		xprintf("    HEVC Constant Frame Rate: %d\n", hevc_desc->GetConstantFrameRate());
		xprintf("    HEVC NALU Length Size:    %d\n", hevc_desc->GetNaluLengthSize());
		xprintf("    HEVC Sequences:\n");

		for (unsigned int i = 0; i < hevc_desc->GetSequences().ItemCount(); i++) 
		{
			const AP4_HvccAtom::Sequence& seq = hevc_desc->GetSequences()[i];
			xprintf("      {\n");
			xprintf("        Array Completeness=%d\n", seq.m_ArrayCompleteness);
			xprintf("        Type=%d", seq.m_NaluType);

			const char* nalu_type_name = AP4_HevcNalParser::NaluTypeName(seq.m_NaluType);
			if (nalu_type_name) 
			{
				xprintf(" (%s)", nalu_type_name);
			}
			xprintf("\n");
			for (unsigned int j = 0; j < seq.m_Nalus.ItemCount(); j++) 
			{
				xprintf("        ");
				ShowData(seq.m_Nalus[j]);
			}
			xprintf("\n      }\n");
		}

		xprintf("    Codecs String: ");
		AP4_String codec;
		hevc_desc->GetCodecString(codec);
		xprintf("%s", codec.GetChars());
		xprintf("\n");
	}

	// Dolby Vision specifics
	AP4_DvccAtom* dvcc = AP4_DYNAMIC_CAST(AP4_DvccAtom, desc->GetDetails().GetChild(AP4_ATOM_TYPE_DVCC));
	if (dvcc) 
	{
		xprintf("    Dolby Vision:\n");
		xprintf("      Version:     %d.%d\n", dvcc->GetDvVersionMajor(), dvcc->GetDvVersionMinor());

		const char* profile_name = AP4_DvccAtom::GetProfileName(dvcc->GetDvProfile());
		if (profile_name) 
		{
			xprintf("      Profile:     %s\n", profile_name);
		}
		else 
		{
			xprintf("      Profile:     %d\n", dvcc->GetDvProfile());
		}
		xprintf("      Level:       %d\n", dvcc->GetDvLevel());
		xprintf("      RPU Present: %s\n", dvcc->GetRpuPresentFlag() ? "true" : "false");
		xprintf("      EL Present:  %s\n", dvcc->GetElPresentFlag() ? "true" : "false");
		xprintf("      BL Present:  %s\n", dvcc->GetBlPresentFlag() ? "true" : "false");
	}

	// VPx Specifics
	if (desc->GetFormat() == AP4_SAMPLE_FORMAT_VP8 ||
		desc->GetFormat() == AP4_SAMPLE_FORMAT_VP9 ||
		desc->GetFormat() == AP4_SAMPLE_FORMAT_VP10) 
	{
		AP4_VpccAtom* vpcc = AP4_DYNAMIC_CAST(AP4_VpccAtom, desc->GetDetails().GetChild(AP4_ATOM_TYPE_VPCC));
		if (vpcc) 
		{
			xprintf("    Profile:                  %d\n", vpcc->GetProfile());
			xprintf("    Level:                    %d\n", vpcc->GetLevel());
			xprintf("    Bit Depth:                %d\n", vpcc->GetBitDepth());
			xprintf("    Chroma Subsampling:       %d\n", vpcc->GetChromaSubsampling());
			xprintf("    Colour Primaries:         %d\n", vpcc->GetColourPrimaries());
			xprintf("    Transfer Characteristics: %d\n", vpcc->GetTransferCharacteristics());
			xprintf("    Matrix Coefficients:      %d\n", vpcc->GetMatrixCoefficients());
			xprintf("    Video Full Range Flag:    %s\n", vpcc->GetVideoFullRangeFlag() ? "true" : "false");

			AP4_String codec;
			vpcc->GetCodecString(desc->GetFormat(), codec);
			xprintf("    Codecs String:            %s", codec.GetChars());
			xprintf("\n");
		}
	}

	// Subtitles
	if (desc->GetType() == AP4_SampleDescription::TYPE_SUBTITLES) 
	{
		AP4_SubtitleSampleDescription* subt_desc = AP4_DYNAMIC_CAST(AP4_SubtitleSampleDescription, desc);
		xprintf("    Subtitles:\n");
		xprintf("       Namespace:       %s\n", subt_desc->GetNamespace().GetChars());
		xprintf("       Schema Location: %s\n", subt_desc->GetSchemaLocation().GetChars());
		xprintf("       Image Mime Type: %s\n", subt_desc->GetImageMimeType().GetChars());
	}

}

void XBento4::ShowData(const AP4_DataBuffer& data)
{
	for (unsigned int i = 0; i < data.GetDataSize(); i++) 
	{
		xprintf("%02x", (unsigned char)data.GetData()[i]);
	}
}

void XBento4::ShowProtectedSampleDescription_Text(AP4_ProtectedSampleDescription& desc, bool verbose)
{
	xprintf("    [ENCRYPTED]\n");
	char coding[5];
	AP4_FormatFourChars(coding, desc.GetFormat());
	xprintf("      Coding:         %s\n", coding);

	AP4_UI32 st = desc.GetSchemeType();
	xprintf("      Scheme Type:    %c%c%c%c\n",
		(char)((st >> 24) & 0xFF),
		(char)((st >> 16) & 0xFF),
		(char)((st >> 8) & 0xFF),
		(char)((st) & 0xFF));
	xprintf("      Scheme Version: %d\n", desc.GetSchemeVersion());
	xprintf("      Scheme URI:     %s\n", desc.GetSchemeUri().GetChars());

	AP4_ProtectionSchemeInfo* scheme_info = desc.GetSchemeInfo();
	if (scheme_info == NULL) 
		return;

	AP4_ContainerAtom* schi = scheme_info->GetSchiAtom();
	if (schi == NULL) 
		return;

	ShowProtectionSchemeInfo_Text(desc.GetSchemeType(), *schi, verbose);
}

void XBento4::ShowProtectionSchemeInfo_Text(AP4_UI32 scheme_type, AP4_ContainerAtom& schi, bool verbose)
{
	if (scheme_type == AP4_PROTECTION_SCHEME_TYPE_IAEC) 
	{
		xprintf("      iAEC Scheme Info:\n");
		AP4_IkmsAtom* ikms = AP4_DYNAMIC_CAST(AP4_IkmsAtom, schi.FindChild("iKMS"));
		if (ikms) 
		{
			xprintf("        KMS URI:              %s\n", ikms->GetKmsUri().GetChars());
		}
		AP4_IsfmAtom* isfm = AP4_DYNAMIC_CAST(AP4_IsfmAtom, schi.FindChild("iSFM"));
		if (isfm) 
		{
			xprintf("        Selective Encryption: %s\n", isfm->GetSelectiveEncryption() ? "yes" : "no");
			xprintf("        Key Indicator Length: %d\n", isfm->GetKeyIndicatorLength());
			xprintf("        IV Length:            %d\n", isfm->GetIvLength());
		}
		AP4_IsltAtom* islt = AP4_DYNAMIC_CAST(AP4_IsltAtom, schi.FindChild("iSLT"));
		if (islt) 
		{
			xprintf("        Salt:                 ");
			for (unsigned int i = 0; i < 8; i++) 
			{
				xprintf("%02x", islt->GetSalt()[i]);
			}
			xprintf("\n");
		}
	}
	else if (scheme_type == AP4_PROTECTION_SCHEME_TYPE_OMA) 
	{
		xprintf("      odkm Scheme Info:\n");
		AP4_OdafAtom* odaf = AP4_DYNAMIC_CAST(AP4_OdafAtom, schi.FindChild("odkm/odaf"));
		if (odaf) 
		{
			xprintf("        Selective Encryption: %s\n", odaf->GetSelectiveEncryption() ? "yes" : "no");
			xprintf("        Key Indicator Length: %d\n", odaf->GetKeyIndicatorLength());
			xprintf("        IV Length:            %d\n", odaf->GetIvLength());
		}
		AP4_OhdrAtom* ohdr = AP4_DYNAMIC_CAST(AP4_OhdrAtom, schi.FindChild("odkm/ohdr"));
		if (ohdr) 
		{
			const char* encryption_method = "";
			switch (ohdr->GetEncryptionMethod()) 
			{
				case AP4_OMA_DCF_ENCRYPTION_METHOD_NULL:
					encryption_method = "NULL";
					break;
				
				case AP4_OMA_DCF_ENCRYPTION_METHOD_AES_CTR:
					encryption_method = "AES-CTR";
					break;
				
				case AP4_OMA_DCF_ENCRYPTION_METHOD_AES_CBC:
					encryption_method = "AES-CBC";
					break;

				default:
					encryption_method = "UNKNOWN";
					break;
			}
			xprintf("        Encryption Method: %s\n", encryption_method);
			xprintf("        Content ID:        %s\n", ohdr->GetContentId().GetChars());
			xprintf("        Rights Issuer URL: %s\n", ohdr->GetRightsIssuerUrl().GetChars());

			const AP4_DataBuffer& headers = ohdr->GetTextualHeaders();
			AP4_Size              data_len = headers.GetDataSize();
			if (data_len) 
			{
				AP4_Byte*      textual_headers_string;
				AP4_Byte*      curr;
				AP4_DataBuffer output_buffer;
				output_buffer.SetDataSize(data_len + 1);
				AP4_CopyMemory(output_buffer.UseData(), headers.GetData(), data_len);
				curr = textual_headers_string = output_buffer.UseData();
				textual_headers_string[data_len] = '\0';
				while (curr < textual_headers_string + data_len) 
				{
					if ('\0' == *curr) 
					{
						*curr = '\n';
					}
					curr++;
				}
				xprintf("        Textual Headers: \n%s\n", (const char*)textual_headers_string);
			}
		}
	}
	else if (scheme_type == AP4_PROTECTION_SCHEME_TYPE_ITUNES) {
		xprintf("      itun Scheme Info:\n");
		AP4_Atom* name = schi.FindChild("name");
		if (name) 
		{
			xprintf("        Name:    ");
			ShowPayload(*name, true);
			xprintf("\n");
		}
		AP4_Atom* user = schi.FindChild("user");
		if (user) 
		{
			xprintf("        User ID: ");
			ShowPayload(*user);
			xprintf("\n");
		}
		AP4_Atom* key = schi.FindChild("key ");
		if (key) 
		{
			xprintf("        Key ID:  ");
			ShowPayload(*key);
			xprintf("\n");
		}
		AP4_Atom* iviv = schi.FindChild("iviv");
		if (iviv) 
		{
			xprintf("        IV:      ");
			ShowPayload(*iviv);
			xprintf("\n");
		}
	}
	else if (
		scheme_type == AP4_PROTECTION_SCHEME_TYPE_MARLIN_ACBC ||
		scheme_type == AP4_PROTECTION_SCHEME_TYPE_MARLIN_ACGK) 
	{
		xprintf("      Marlin IPMP ACBC/ACGK Scheme Info:\n");
		AP4_NullTerminatedStringAtom* octopus_id = AP4_DYNAMIC_CAST(AP4_NullTerminatedStringAtom, schi.FindChild("8id "));
		if (octopus_id) 
		{
			xprintf("        Content ID: %s\n", octopus_id->GetValue().GetChars());
		}
	}

	if (verbose) 
	{
		xprintf("    Protection System Details:\n");
		AP4_ByteStream* output = NULL;
		AP4_FileByteStream::Create("-stdout", AP4_FileByteStream::STREAM_MODE_WRITE, output);
		AP4_PrintInspector inspector(*output, 4);
		schi.Inspect(inspector);
		output->Release();
	}
}

void XBento4::ShowPayload(AP4_Atom& atom, bool ascii)
{
	AP4_UI64 payload_size = atom.GetSize() - 8;
	if (payload_size <= 1024) 
	{
		AP4_MemoryByteStream* payload = new AP4_MemoryByteStream();
		atom.Write(*payload);
		if (ascii) 
		{
			// ascii
			payload->WriteUI08(0); // terminate with a NULL character
			xprintf("%s", (const char*)payload->GetData() + atom.GetHeaderSize());
		}
		else 
		{
			// hex
			for (unsigned int i = 0; i < payload_size; i++) 
			{
				xprintf("%02x", (unsigned char)payload->GetData()[atom.GetHeaderSize() + i]);
			}
		}
		payload->Release();
	}
}

void XBento4::ShowMpegAudioSampleDescription(AP4_MpegAudioSampleDescription& mpeg_audio_desc)
{
	AP4_MpegAudioSampleDescription::Mpeg4AudioObjectType object_type =
		mpeg_audio_desc.GetMpeg4AudioObjectType();
	const char* object_type_string = AP4_MpegAudioSampleDescription::GetMpeg4AudioObjectTypeString(object_type);
	AP4_String codec_string;
	mpeg_audio_desc.GetCodecString(codec_string);

	xprintf("    Codecs String: %s\n", codec_string.GetChars());
	xprintf("    MPEG-4 Audio Object Type: %d (%s)\n", object_type, object_type_string);

	// Decoder Specific Info
	const AP4_DataBuffer& dsi = mpeg_audio_desc.GetDecoderInfo();
	if (dsi.GetDataSize()) 
	{
		AP4_Mp4AudioDecoderConfig dec_config;
		AP4_Result result = dec_config.Parse(dsi.GetData(), dsi.GetDataSize());
		if (AP4_SUCCEEDED(result)) 
		{
			xprintf("    MPEG-4 Audio Decoder Config:\n");
			xprintf("      Sampling Frequency: %d\n", dec_config.m_SamplingFrequency);
			xprintf("      Channels: %d\n", dec_config.m_ChannelCount);
			if (dec_config.m_Extension.m_ObjectType) {
				object_type_string = AP4_MpegAudioSampleDescription::GetMpeg4AudioObjectTypeString(
					dec_config.m_Extension.m_ObjectType);

				xprintf("      Extension:\n");
				xprintf("        Object Type: %s\n", object_type_string);
				xprintf("        SBR Present: %s\n", dec_config.m_Extension.m_SbrPresent ? "yes" : "no");
				xprintf("        PS Present:  %s\n", dec_config.m_Extension.m_PsPresent ? "yes" : "no");
				xprintf("        Sampling Frequency: %d\n", dec_config.m_Extension.m_SamplingFrequency);
			}
		}
	}
}

void XBento4::ScanMedia(AP4_Movie& movie, AP4_Track& track, AP4_ByteStream& stream, MediaInfo& info)
{
	AP4_UI64 total_size = 0;
	AP4_UI64 total_duration = 0;

	AP4_UI64 position;
	stream.Tell(position);
	stream.Seek(0);
	AP4_LinearReader reader(movie, &stream);
	reader.EnableTrack(track.GetId());

	info.sample_count = 0;

	AP4_Sample sample;
	if (movie.HasFragments())
	{
		AP4_DataBuffer sample_data;
		for (unsigned int i = 0; ; i++) 
		{
			AP4_UI32 track_id = 0;
			AP4_Result result = reader.ReadNextSample(sample, sample_data, track_id);
			if (AP4_SUCCEEDED(result)) 
			{
				total_size += sample.GetSize();
				total_duration += sample.GetDuration();
				++info.sample_count;
			}
			else 
			{
				break;
			}
		}
	}
	else 
	{
		info.sample_count = track.GetSampleCount();
		for (unsigned int i = 0; i < track.GetSampleCount(); i++)
		{
			if (AP4_SUCCEEDED(track.GetSample(i, sample)))
			{
				total_size += sample.GetSize();
			}
		}
		total_duration = track.GetMediaDuration();
	}
	info.duration = total_duration;

	double duration_ms = (double)AP4_ConvertTime(total_duration, track.GetMediaTimeScale(), 1000);
	if (duration_ms) 
	{
		info.bitrate = 8.0*1000.0*(double)total_size / duration_ms;
	}
	else 
	{
		info.bitrate = 0.0;
	}
}
