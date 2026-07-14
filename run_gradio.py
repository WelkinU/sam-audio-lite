from sam_audio_lite._compat import patch_windows_asyncio, prepare_windows_dll_paths

# Must run before Gradio/uvicorn start the asyncio event loop.
patch_windows_asyncio()
prepare_windows_dll_paths()

from sam_audio_lite.app import main

if __name__ == "__main__":
    main()
