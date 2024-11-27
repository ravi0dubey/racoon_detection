#Kubeflow setup
import kfp
from kfp import dsl
from kfp import compiler
import os

# Load the component from the YAML file.
#
#  If we need to run the code in virtual env we need to use below commented code
video_frame_extractor_op = kfp.components.load_component_from_file('component.yaml')

#  If we need to debug/run the code in terminal env we need to use below code
# current_dir = os.path.dirname(os.path.abspath(__file__))
# component_path = os.path.join(current_dir, 'component.yaml')
# video_frame_extractor_op = kfp.components.load_component_from_file(component_path)

@dsl.pipeline(name='video-frame-extraction-pipeline',description='A pipeline that extracts frames from input videos using a custom container.')
def pipeline( input_source: str, output_path: str, frame_rate: int = 1,):
    video_frame_extractor_task = video_frame_extractor_op(input_source=input_source,output_path=output_path,frame_rate=frame_rate)

if __name__ == '__main__':
    compiler.Compiler().compile(pipeline_func=pipeline,package_path='video_frame_extraction_pipeline.json')