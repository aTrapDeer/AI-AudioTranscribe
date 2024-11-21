# Copyright 2023-2024 Deepgram SDK contributors. All Rights Reserved.
# Use of this source code is governed by a MIT license that can be found in the LICENSE file.
# SPDX-License-Identifier: MIT

import os
from dotenv import load_dotenv
import logging
from deepgram.utils import verboselogs
from datetime import datetime
import httpx
import openai

from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    PrerecordedOptions,
    FileSource,
)

load_dotenv()
AUDIO_FOLDER = "Audio"



def create_summary(transcript_file):
    """
    Create a summary of the transcript using OpenAI API and save it
    """
    try:
        # Read the transcript file
        with open(transcript_file, 'r') as file:
            transcript_text = file.read()

        # Configure OpenAI
        openai.api_key = os.getenv('OPENAI_API_KEY')
        
        # Create the summary using OpenAI
        response = openai.chat.completions.create(
            model="gpt-4o-mini",  # or "gpt-3.5-turbo" if preferred
            messages=[
                {
                    "role": "system",
                    "content": "Create a clear, organized summary of this transcript. Break it down into main topics and action items."
                },
                {
                    "role": "user",
                    "content": transcript_text
                }
            ],
            max_tokens=1000
        )
        
        # Extract summary
        summary = response.choices[0].message.content
        # Create summary filename in Summarization folder based on original transcript filename
        base_path = os.path.splitext(os.path.basename(transcript_file))[0]
        summary_dir = "Summarization"
        os.makedirs(summary_dir, exist_ok=True)
        summary_filename = os.path.join(summary_dir, f"Summary_{base_path}.txt")
        
        # Save summary to file
        with open(summary_filename, 'w') as file:
            file.write(summary)
            
        print(f"Summary saved to {summary_filename}")
        return summary
    except Exception as e:
        print(f"Error creating summary: {e}")
        return None

def process_audio_file(audio_file_path, deepgram):
    """
    Process a single audio file - transcribe and summarize
    """
    try:
        print(f"\nProcessing: {audio_file_path}")
        
        # Read the audio file
        with open(audio_file_path, "rb") as file:
            buffer_data = file.read()

        payload: FileSource = {
            "buffer": buffer_data,
        }

        options: PrerecordedOptions = PrerecordedOptions(
            model="nova-2",
            smart_format=True,
            utterances=True,
            punctuate=True,
            diarize=True,
        )

        before = datetime.now()
        response = deepgram.listen.rest.v("1").transcribe_file(
            payload, options, timeout=httpx.Timeout(300.0, connect=10.0)
        )
        after = datetime.now()

        # Extract and format transcripts
        transcripts = []
        for result in response.results.utterances:
            transcripts.append(result.transcript)
        
        # Join transcripts with newlines
        full_transcript = "\n".join(transcripts)
        
        # Create filename based on original audio filename
        base_filename = os.path.splitext(os.path.basename(audio_file_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save transcript
        transcript_dir = "Transcription"
        os.makedirs(transcript_dir, exist_ok=True)
        transcript_filename = os.path.join(transcript_dir, f"transcript_{base_filename}_{timestamp}.txt")
        
        with open(transcript_filename, "w") as transcript_file:
            transcript_file.write(full_transcript)

        print(f"Transcript saved to {transcript_filename}")
        print(f"Time taken: {(after - before).seconds} seconds")

        # Create summary
        if os.path.exists(transcript_filename):
            create_summary(transcript_filename)

        return True

    except Exception as e:
        print(f"Error processing {audio_file_path}: {e}")
        return False

def main():
    try:
        # Create Deepgram client
        config: DeepgramClientOptions = DeepgramClientOptions(
            verbose=verboselogs.SPAM,
        )
        deepgram: DeepgramClient = DeepgramClient("", config)

        # Process all audio files in the folder
        successful_files = 0
        failed_files = 0
        
        # Get list of supported audio file extensions
        supported_extensions = ['.mp3', '.wav', '.m4a', '.ogg', '.flac']
        
        # Process each audio file
        for filename in os.listdir(AUDIO_FOLDER):
            if any(filename.lower().endswith(ext) for ext in supported_extensions):
                audio_file_path = os.path.join(AUDIO_FOLDER, filename)
                
                if process_audio_file(audio_file_path, deepgram):
                    successful_files += 1
                else:
                    failed_files += 1

        print(f"\nProcessing complete!")
        print(f"Successfully processed: {successful_files} files")
        print(f"Failed to process: {failed_files} files")

    except Exception as e:
        print(f"Main exception: {e}")

if __name__ == "__main__":
    main()