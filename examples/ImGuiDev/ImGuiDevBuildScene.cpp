/**************************************************************
** ImGuiDevBuildScene.cpp
**
** ImGui is a 3rd-party GUI library with tremendous appeal for
** me:  I REALLY don't want to write a GUI layer, because
** writing GUI widgets is way too tedious. ImGui looks like
** it can be made to be pretty enough for professional looking
** UI experiences, which I care about.
**************************************************************/
#include "ExampleXGL.h"
#include "xfifo.h"
#include "xsqlite.h"
#include "xbento4_class.h"
#include "xglvcrcontrols.h"
#include "xh264.h"
#include "fifotest.h"
#include "xlog.h"

namespace {
	XLOG_DECLARE("ImGuiDevBuildScene");

	XGLVcrControlsGui* xig{ nullptr };
	Xsqlite* xdb{ nullptr };
	XBento4* xb4{ nullptr };
	Xh264Decoder* xdecoder{ nullptr };

	XFifoTest::Tester* fifoTester{ nullptr };

	// can't use alignas() operator on heap objects, so make this a
	// file-scope global for now.
	//
	// Alternative is to overload new (& delete?) operators to produce
	// the desired alignment at runtime.  Do that later.
	XFifo xf(0x100000);

	// defined/declared in JM ldecode project
	extern "C" void allocate_p_dec_pic(
		VideoParameters *p_Vid,
		DecodedPicList *pDecPic,
		StorablePicture *p,
		int iLumaSize,
		int iFrameSize,
		int iLumaSizeX,
		int iLumaSizeY,
		int iChromaSizeX,
		int iChromaSizeY
	);
}


void DisplayFrame(VideoParameters* p_Vid, StorablePicture* p, int p_out) 
{
	Xh264Decoder* pDecoder = (Xh264Decoder*)p_Vid->p_Inp->p_ctx;
	InputParameters *p_Inp = p_Vid->p_Inp;
	DecodedPicList *pDecPic;

	static const int SubWidthC[4] = { 1, 2, 2, 1 };
	static const int SubHeightC[4] = { 1, 2, 1, 1 };

	int crop_left, crop_right, crop_top, crop_bottom;
	int symbol_size_in_bytes = ((p_Vid->pic_unit_bitsize_on_disk + 7) >> 3);
	unsigned char *buf;
	int iLumaSize, iFrameSize;
	int iLumaSizeX, iLumaSizeY;
	int iChromaSizeX, iChromaSizeY;

	if (p->non_existing)
		return;

	// should this be done only once?
	if (p->frame_cropping_flag)
	{
		crop_left = SubWidthC[p->chroma_format_idc] * p->frame_crop_left_offset;
		crop_right = SubWidthC[p->chroma_format_idc] * p->frame_crop_right_offset;
		crop_top = SubHeightC[p->chroma_format_idc] * (2 - p->frame_mbs_only_flag) * p->frame_crop_top_offset;
		crop_bottom = SubHeightC[p->chroma_format_idc] * (2 - p->frame_mbs_only_flag) * p->frame_crop_bottom_offset;
	}
	else
	{
		crop_left = crop_right = crop_top = crop_bottom = 0;
	}
	iChromaSizeX = p->size_x_cr - p->frame_crop_left_offset - p->frame_crop_right_offset;
	iChromaSizeY = p->size_y_cr - (2 - p->frame_mbs_only_flag) * p->frame_crop_top_offset - (2 - p->frame_mbs_only_flag) * p->frame_crop_bottom_offset;
	iLumaSizeX = p->size_x - crop_left - crop_right;
	iLumaSizeY = p->size_y - crop_top - crop_bottom;
	iLumaSize = iLumaSizeX * iLumaSizeY * symbol_size_in_bytes;
	iFrameSize = (iLumaSizeX * iLumaSizeY + 2 * (iChromaSizeX * iChromaSizeY)) * symbol_size_in_bytes; //iLumaSize*iPicSizeTab[p->chroma_format_idc]/2;

	// We need to further cleanup this function
	if (p_out == -1)
		return;

	// KS: this buffer should actually be allocated only once, but this is still much faster than the previous version
	pDecPic = get_one_avail_dec_pic_from_list(p_Vid->pDecOuputPic, 0, 0);
	if ((pDecPic->pY == NULL) || (pDecPic->iBufSize < iFrameSize))
		allocate_p_dec_pic(p_Vid, pDecPic, p, iLumaSize, iFrameSize, iLumaSizeX, iLumaSizeY, iChromaSizeX, iChromaSizeY);

#if (MVC_EXTENSION_ENABLE)
	{
		pDecPic->bValid = 1;
		pDecPic->iViewId = p->view_id >= 0 ? p->view_id : -1;
	}
#else
	pDecPic->bValid = 1;
#endif

	pDecPic->iPOC = p->frame_poc;

	if (NULL == pDecPic->pY)
	{
		XLOG("write_out_picture: buf");
		return;
	}

	buf = (pDecPic->bValid == 1) ? pDecPic->pY : pDecPic->pY + iLumaSizeX * symbol_size_in_bytes;
	p_Vid->img2buf(p->imgY, buf, p->size_x, p->size_y, symbol_size_in_bytes, crop_left, crop_right, crop_top, crop_bottom, pDecPic->iYBufStride);

	int ySize = (p->size_y - crop_bottom - crop_top)*(p->size_x - crop_right - crop_left) * symbol_size_in_bytes;
	memcpy(pDecoder->yuvBuffer, buf, ySize);

	if (p->chroma_format_idc != YUV400)
	{
		crop_left = p->frame_crop_left_offset;
		crop_right = p->frame_crop_right_offset;
		crop_top = (2 - p->frame_mbs_only_flag) * p->frame_crop_top_offset;
		crop_bottom = (2 - p->frame_mbs_only_flag) * p->frame_crop_bottom_offset;

		int uvSize = (p->size_y_cr - crop_bottom - crop_top)*(p->size_x_cr - crop_right - crop_left)* symbol_size_in_bytes;

		buf = (pDecPic->bValid == 1) ? pDecPic->pU : pDecPic->pU + iChromaSizeX * symbol_size_in_bytes;
		p_Vid->img2buf(p->imgUV[0], buf, p->size_x_cr, p->size_y_cr, symbol_size_in_bytes, crop_left, crop_right, crop_top, crop_bottom, pDecPic->iUVBufStride);
		memcpy(pDecoder->yuvBuffer + ySize, buf, uvSize);

		buf = (pDecPic->bValid == 1) ? pDecPic->pV : pDecPic->pV + iChromaSizeX * symbol_size_in_bytes;
		p_Vid->img2buf(p->imgUV[1], buf, p->size_x_cr, p->size_y_cr, symbol_size_in_bytes, crop_left, crop_right, crop_top, crop_bottom, pDecPic->iUVBufStride);
		memcpy(pDecoder->yuvBuffer + ySize + uvSize, buf, uvSize);
	}

	//free(buf);
	if (p_out >= 0)
		pDecPic->bValid = 0;

	//  fsync(p_out);
};

void ExampleXGL::BuildScene() {
	std::string dbPath = pathToAssets + "/assets/dbTest.sq3";

	try 
	{
		//xdb = new Xsqlite(dbPath);
		//xig = new XGLVcrControlsGui();
		//AddShape("shaders/yuv", [&]() { xdecoder = new Xh264Decoder(); return xdecoder; });
		AddShape("shaders/000-simple", [&]() { fifoTester = new XFifoTest::Tester(&xf); return fifoTester; });
	}
	catch (std::exception e)
	{
		XLOG("Caught exception: %s", e.what());
	}

	if (xdb)
	{
		xdb->AddCallback([&](int argc, char**argv, char** columnNames)
		{
			Xsqlite::KeyValueList kl;

			for (int i = 0; i < argc; i++)
				kl.push_back({ columnNames[i], argv[i] });

			std::string row;
			for (Xsqlite::KeyValue k : kl)
				row += k.first + ": " + k.second + ", ";

			XLOG("%s", row.c_str());

			return 0;
		});

		std::string  sql = "DROP TABLE IF EXISTS Cars;"
			"CREATE TABLE Cars(Id INT, Name TEXT, Price INT);"
			"INSERT INTO Cars VALUES(1, 'Audi', 52642);"
			"INSERT INTO Cars VALUES(2, 'Mercedes', 57127);"
			"INSERT INTO Cars VALUES(3, 'Skoda', 9000);"
			"INSERT INTO Cars VALUES(4, 'Volvo', 29000);"
			"INSERT INTO Cars VALUES(5, 'Bentley', 350000);"
			"INSERT INTO Cars VALUES(6, 'Citroen', 21000);"
			"INSERT INTO Cars VALUES(7, 'Hummer', 41400);"
			"INSERT INTO Cars VALUES(8, 'Volkswagen', 21600);";

		xdb->Execute(sql);

		xdb->Execute("SELECT name FROM sqlite_master WHERE type = 'table';");
		xdb->Execute("SELECT * FROM Cars;");
	}
	else
		XLOG("xdb object not available");

	//AddShape("shaders/yuv", [&]() { xb4 = new XBento4("H:/Hero6/GH010171.mp4"); return xb4; });
	//glm::mat4 scale = glm::scale(glm::mat4(), { 16,9,0 });
	//xb4->model = scale;

	if (xig)
	{
		menuFunctions.push_back(([&]() {
			if (ImGui::Begin("VCR Controls", &xig->vcrWindow))
			{
				ImGui::SliderInt("Frame#", &xig->frameNum, 0, 1000);
				//if (ImGui::SliderInt("Frame", &xig->frameNum, 0, xb4->GetNumFrames() - 1))
				//{
					//xb4->SeekToFrame(xig->frameNum);
				//}
			}
			ImGui::End();
		}));
	}
	else
		XLOG("xig object not available");

	if (xdecoder)
	{
		glm::mat4 scale = glm::scale(glm::mat4(), { 16,9,0 });
		xdecoder->model = scale;

		xdecoder->AddCallback(DisplayFrame);
		xdecoder->Start();
	}
	else
		XLOG("xdecoder object not available");

	XLOG("xf.Available(): %llu", xf.Available());

	if (fifoTester)
	{
		XLOG("fifoTester exists.");
		fifoTester->Start();
	}
	else
		XLOG("fifoTester does not exist.");
}
