#include <stage.hh>

using namespace Stg;

struct ModelRobot
{
  ModelPosition* pos;
  ModelRanger* laser;
  Pose resetPose;
  int avoidcount, randcount;
};

ModelRobot* robot;
usec_t stgSpeedTime;

void stgLaserCB( Model* mod, ModelRobot* robot)
{
  std::cout << "Laser CB... " << std::endl;
}

extern "C" int Init( Model* mod )
{ 
  int argc = 0;
  char** argv;
  
  robot = new ModelRobot;
  robot->pos = (ModelPosition*) mod;
  
  robot->laser = (ModelRanger*)mod->GetChild("ranger:0");
  robot->laser->AddCallback( Model::CB_UPDATE, (model_callback_t)stgLaserCB, robot);
  robot->laser->Subscribe();
  return 0; //ok
}
