#include "ExampleXGL.h"
#include "xglimgui.h"

bool testWindow{ true };
int readPoolSize{ 0 };

void ExampleXGL::BuildGUI() {
	XGLImGui* gm;
/*
	menuFunctions.push_back(([&]() {
		if (ImGui::Begin("Gravity Controls", &testWindow))
		{
			ImGui::SliderInt("Read Pool Size", &readPoolSize, 0, 64);
			ImGui::SliderInt("Read Buffer Size", &fifoTestControls->readBufferSize, 0, 0x100000);
			ImGui::SliderInt("Write Pool Size", &fifoTestControls->writePoolSize, 0, 64);
			ImGui::SliderInt("Write Buffer Size", &fifoTestControls->writeBufferSize, 0, 0x100000);

			fifoTestControls->isRunning = fifoTester->IsRunning();

			if (ImGui::Checkbox("Running", &fifoTestControls->isRunning))
			{
				XLOG(XLDebug, "isRunning changed to: %s", (fifoTestControls->isRunning) ? "Running" : "Stopped");
			}
			if (!fifoTestControls->isRunning)
			{
				if (ImGui::Button("Start", { 60,24 }))
				{
					XLOG(XLDebug, "Start button clicked");
					fifoTester->Start();
				}
			}
			else
			{
				if (ImGui::Button("Stop", { 60,24 }))
				{
					XLOG(XLDebug, "Stop button clicked");
					fifoTester->Stop();
				}
			}
			ImGui::SameLine(80.0f);
			if (ImGui::Checkbox("Reader Enabled", &fifoTestControls->isReading))
			{
				XLOG(XLDebug, "\"Reader Enabled\" to: %s", (fifoTestControls->isReading) ? "Enabled" : "Disabled");
			}
			ImGui::SameLine(280.0f);
			if (ImGui::Checkbox("Writer Enabled", &fifoTestControls->isWriting))
			{
				XLOG(XLDebug, "\"Writer Enabled\" to: %s", (fifoTestControls->isWriting) ? "Enabled" : "Disabled");
			}

			float fraction = (float)xf.Used() / (float)xf.Capacity();
			ImGui::ProgressBar(fraction, { 400, 20 });

		}
		ImGui::End();
	}));

	AddGuiShape("shaders/ortho", [&]() { gm = new XGLImGui(); return gm; });
*/
	return;
}
