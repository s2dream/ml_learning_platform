# ml_learning_platform
skeleton codes for learning ml models


## Example Instruction
- Dummy Execution: python main.py --task dummy_train --model dummy_classifier --optimizer adamw --dataloader dummy

## How to add a module
1. configuration, dataproc/data_loader, model, task의 하위에 module 이름의 directory를 생성한다.
2. 각각에 해당하는 pytorch 기반 모듈을 구현한다.
3. ./argument_description.json 파일에 모듈명을 key로, module file path를 value롤 각각 항목에 추가한다. key는 main.py 실행 시 argument name으로 받아서 동적으로 instantiation 후에 연결함. 
   1. task, model, configuration, optimizer, dataloader 항목이 있음.
   2. task의 경우 train과 test로 분리해서 추가하는 것을 추천 (예: dummy_traiin, dummy_test)
   3. 2번에 따라, configuration 역시 분리해서 추가하는 것을 추천 (configuration의 경우 train과 test 각각을 위한 setting method를 만들고 task에서 상황에 맞는 method를 호출해서 설정 가능)
   4. model명의 경우 조금 더 자세하게 기술하는 것을 추천 (에: transformer encoder, relation classifier, etc)
   5. optimizer의 경우 torch의 기본 모듈을 활용한다.
4. python main.py --task \<task name> --model \<model name> --optimizer \<optimizer> --dataloader \<dataloader>
5. script를 만들어두기를 추천.


## Architecture (Class Diagram)
![Architecture (Class Diagram)](readme_image/architecture(class_diagram).png)