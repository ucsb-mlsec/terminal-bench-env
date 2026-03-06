#include <iostream>
#include   <vector>
#include<string>
#include <memory>
#include     <algorithm>

namespace   Engine{
class    GameEngine
{
  public:
      GameEngine( ):   m_running(   false  ),m_frameCount(0){}
    void initialize(std::string const&name,int width,int    height){
            m_name=name;m_width=width;m_height=height;
        std::cout<<"Initializing engine: "<<m_name<<" with resolution "<<m_width<<"x"<<m_height<<std::endl;
            m_running = true;
    }

void update( double   deltaTime  )
{
        if(   !m_running   ){return;}
    m_frameCount++;
        for(  int i = 0 ; i< m_entities.size( ) ;++i ){
            m_entities[  i  ]->update(   deltaTime   );
        }
            processPhysics(deltaTime);
}

        void render( )   {
if(!m_running)return;
                std::cout << "Rendering frame " << m_frameCount << std::endl;
        for(   auto   &entity : m_entities  )
        {
entity->render(  );
            }
    }

    void addEntity(  std::shared_ptr<  Entity >   entity  ){m_entities.push_back(entity);}

        int*   getFrameCountPointer( ){return &m_frameCount;}
    int  &  getFrameCountReference(){return m_frameCount;}

  private:
        void processPhysics(  double dt  )
    {
switch(   m_physicsMode   ){
case 0:
    applyGravity( dt );
        break;
            case 1:{applyWind(dt);break;}
                case 2:
                        {
                    applyBothForces(  dt  );
                            break;
        }
    default:break;}}

void applyGravity(double  dt){for(auto&  e:m_entities){e->applyForce(0,-9.81*dt);}}
            void applyWind(  double dt  )
        {
    for(  auto &e : m_entities  ){e->applyForce(5.0*dt,0);}}

        void applyBothForces(   double dt   ){applyGravity( dt );applyWind(dt);}

    std::string m_name;
        int m_width,m_height;
    bool   m_running;
            int m_frameCount;
int m_physicsMode;
    std::vector<  std::shared_ptr<Entity>  > m_entities;
};

    class   Entity{
public:
virtual void update(  double  dt  )=0;
        virtual void render()=0;
    virtual void applyForce(  double  x,double   y  )=0;
            virtual ~Entity(  ){}
};
}