extern crate nalgebra as na;

use gdnative::prelude::*;

use na::{Point2, Vector2, Isometry2};
use ncollide2d::shape::{ShapeHandle, Polyline, ConvexPolygon};
use nphysics2d::force_generator::DefaultForceGeneratorSet;
use nphysics2d::joint::DefaultJointConstraintSet;
use nphysics2d::object::{BodyPartHandle, ColliderDesc, DefaultBodySet, DefaultColliderSet, RigidBodyDesc, BodyStatus, DefaultColliderHandle};
use nphysics2d::world::{DefaultGeometricalWorld, DefaultMechanicalWorld};
use salva2d::coupling::{ColliderCouplingSet, CouplingMethod};
use salva2d::object::{Fluid, Boundary};
use salva2d::solver::{ArtificialViscosity, IISPHSolver};
use salva2d::LiquidWorld;
use ncollide2d::bounding_volume::HasBoundingVolume;

#[derive(NativeClass)]
#[inherit(Node)]
struct Physics {
    bodies: DefaultBodySet<f32>,
    colliders: DefaultColliderSet<f32>,
    objects: Vec<gdnative::api::Polygon2D>,
    mechanical_world: DefaultMechanicalWorld<f32>,
    geometrical_world: DefaultGeometricalWorld<f32>,
    liquid_world: LiquidWorld<f32>,
    coupling_set: ColliderCouplingSet<f32, DefaultColliderHandle>,
    joint_constraints: DefaultJointConstraintSet<f32>,
    force_generators: DefaultForceGeneratorSet<f32>,
    particle_rad: f32,
}

#[methods]
impl Physics {

    fn new(_owner: &Node) -> Self {
        Physics {
            bodies: DefaultBodySet::new(),
            colliders: DefaultColliderSet::new(),
            objects: Vec::new(),
            mechanical_world: DefaultMechanicalWorld::new(Vector2::new(0.0, 9.81)),
            geometrical_world: DefaultGeometricalWorld::new(),
            liquid_world: LiquidWorld::new(IISPHSolver::<f32>::new(), 3.0, 2.0),
            coupling_set: ColliderCouplingSet::new(),
            joint_constraints: DefaultJointConstraintSet::new(),
            force_generators: DefaultForceGeneratorSet::new(),
            particle_rad: 3.0
        }
    }

    #[export]
    fn add_rigid_body(&mut self, owner: &Node, position: gdnative::core_types::Vector2, polygon: gdnative::core_types::Vector2Array, is_static: bool) {
        let mut status = BodyStatus::Dynamic;
        if is_static {
            status = BodyStatus::Static;
        }
        let rb = RigidBodyDesc::new().status(status).
            mass(10.0).
            angular_inertia(1.0).
            rotation(0.0).
            translation(Vector2::new(position.x, position.y)).build();

        let rb_handle = self.bodies.insert(rb);

        // Build the collider.
        let geom = ShapeHandle::new(Self::convert_polygon2(polygon));
        let geom_sample =
            salva2d::sampling::shape_surface_ray_sample(&*geom, self.particle_rad).unwrap();
        let co = ColliderDesc::new(geom)
            //.margin(3.0)
            .density(1.0)
            .build(BodyPartHandle(rb_handle, 0));
        let co_handle = self.colliders.insert(co);
        let bo_handle = self.liquid_world.add_boundary(Boundary::new(Vec::new()));
        self.coupling_set.register_coupling(
            bo_handle,
            co_handle,
            CouplingMethod::StaticSampling(geom_sample),
        );
    }

    #[export]
    fn add_liquid(&mut self, owner: &Node, droplets: gdnative::core_types::Vector2Array) {
        let mut points = Vec::new();

        for drop in droplets.read().iter() {
            points.push(Point2::new(drop.x, drop.y));
        }

        let viscosity = ArtificialViscosity::new(0.5, 0.0);
        let mut fluid = Fluid::new(points, self.particle_rad, 1.0);
        fluid.nonpressure_forces.push(Box::new(viscosity.clone()));
        self.liquid_world.add_fluid(fluid);
    }

    #[export]
    fn get_liquid(&mut self, owner: &Node) -> Vector2Array {
        let mut droplets = Vector2Array::new();
        for (i, fluid) in  self.liquid_world.fluids().iter() {
            for droplet in &fluid.positions {
                droplets.push(gdnative::core_types::Vector2::new(droplet.x, droplet.y));
            }
        }
        return droplets;
    }

    #[export]
    fn get_polygons(&mut self, owner: &Node) -> Vector3Array {
        let mut polygons = Vector3Array::new();
        for (i, polygon) in self.colliders.iter() {
            let position: &Isometry2<f32> = polygon.position();
            polygons.push(Vector3::new(position.translation.x, position.translation.y, position.rotation.angle()))
        }
        return polygons;
    }


    fn convert_polygon(polygon: gdnative::core_types::Vector2Array) -> Polyline<f32> {
        let mut points = Vec::new();
        for point in polygon.read().iter() {
            points.push(Point2::new(point.x, point.y));
        }
        return Polyline::new(points, None);
    }

    fn convert_polygon2(polygon: gdnative::core_types::Vector2Array) -> ConvexPolygon<f32> {
        let mut points = Vec::new();
        for point in polygon.read().iter() {
            points.push(Point2::new(point.x, point.y));
        }
        return ConvexPolygon::try_from_points(&points).unwrap();
    }

    #[export]
    fn _process(&mut self, owner: &Node, delta: f32) {
        self.mechanical_world.step(
            &mut self.geometrical_world,
            &mut self.bodies,
            &mut self.colliders,
            &mut self.joint_constraints,
            &mut self.force_generators,
        );


        let dt = self.mechanical_world.timestep();
        let gravity = &self.mechanical_world.gravity;
        self.liquid_world.step_with_coupling(
            dt,
            gravity,
            &mut self.coupling_set
                .as_manager_mut(&self.colliders, &mut self.bodies),
        );
    }
}


fn init(handle: InitHandle) {
    handle.add_class::<Physics>();
}

godot_gdnative_init!();
godot_nativescript_init!(init);
godot_gdnative_terminate!();
