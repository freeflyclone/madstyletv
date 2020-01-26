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
#pragma once

#include "xbento4.h"
#include <string>

typedef struct {
	AP4_UI64 sample_count;
	AP4_UI64 duration;
	double   bitrate;
} MediaInfo;


class XBento4 : public XGLTexQuad, public XThread {
public:
	XBento4();
	XBento4(std::string);
	~XBento4();

	void Run();

	void ShowFileInfo(AP4_File&);

	void ShowMovieInfo(AP4_Movie&);

	void ShowTracks(
		AP4_Movie&, 
		AP4_List<AP4_Track>&, 
		AP4_ByteStream&, 
		bool,
		bool,
		bool,
		bool
	);

	void ShowTrackInfo(
		AP4_Movie&,
		AP4_Track&,
		AP4_ByteStream&,
		bool,
		bool,
		bool,
		bool
	);

	void ScanMedia(
		AP4_Movie&,
		AP4_Track&,
		AP4_ByteStream&,
		MediaInfo&
	);

	void ShowSampleDescription(
		AP4_SampleDescription&,
		bool
	);

	void ShowProtectedSampleDescription_Text(
		AP4_ProtectedSampleDescription& desc,
		bool verbose
	);

	void ShowProtectionSchemeInfo_Text(
		AP4_UI32 scheme_type,
		AP4_ContainerAtom& schi,
		bool verbose
	);

	void ShowPayload(
		AP4_Atom& atom,
		bool ascii = false
	);

	void ShowMpegAudioSampleDescription(
		AP4_MpegAudioSampleDescription& mpeg_audio_desc
	);

	void ShowData(
		const AP4_DataBuffer& data
	);

	void ShowSample(AP4_Track&,
		AP4_Sample&,
		AP4_DataBuffer&,
		unsigned int,
		bool,
		bool,
		AP4_AvcSampleDescription*
	);

	void ShowAvcInfo(
		const AP4_DataBuffer& sample_data,
		AP4_AvcSampleDescription* avc_desc
	);

	unsigned int ReadGolomb(
		AP4_BitStream& bits
	);


private:
	std::string filename;
	AP4_ByteStream* input{ nullptr };
};