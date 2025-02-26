import kfp
from kfp import dsl
from kfp import compiler
import os

# Load the components
video_frame_extractor_op = kfp.components.load_component_from_file('component.yaml')
unique_frame_extractor_op = kfp.components.load_component_from_file('unique_component.yaml')

@dsl.pipeline(name='advanced-video-frame-extraction-pipeline', description='A pipeline that extracts frames using different strategies')
def advanced_pipeline(
    input_source: str, 
    output_path: str, 
    frame_rate: int = 1,
    similarity_threshold: float = 0.95
):
    # Standard frame extraction
    video_frame_extractor_task = video_frame_extractor_op(
        input_source=input_source,
        output_path=output_path + '/standard_frames',
        frame_rate=frame_rate
    )
    
    # Unique frame extraction
    unique_frame_extractor_task = unique_frame_extractor_op(
        input_source=input_source,
        output_path=output_path + '/unique_frames',
        frame_rate=frame_rate,
        similarity_threshold=similarity_threshold
    )

if __name__ == '__main__':
    compiler.Compiler().compile(pipeline_func=advanced_pipeline, package_path='advanced_video_frame_extraction_pipeline.json')



