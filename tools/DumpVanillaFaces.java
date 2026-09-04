import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.Map;

import net.minecraft.SharedConstants;
import net.minecraft.core.BlockPos;
import net.minecraft.core.Direction;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.server.Bootstrap;
import net.minecraft.world.level.EmptyBlockGetter;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.FireBlock;
import net.minecraft.world.level.block.SupportType;
import net.minecraft.world.level.block.WallBlock;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.phys.shapes.BooleanOp;
import net.minecraft.world.phys.shapes.Shapes;
import net.minecraft.world.phys.shapes.VoxelShape;

/** Emits exact per-state connectivity metadata used by the Python resolver. */
public final class DumpVanillaFaces {
    private DumpVanillaFaces() {}

    public static void main(String[] args) throws ReflectiveOperationException {
        SharedConstants.tryDetectVersion();
        Bootstrap.bootStrap();
        Method canBurn = FireBlock.class.getDeclaredMethod("canBurn", BlockState.class);
        canBurn.setAccessible(true);
        FireBlock fire = (FireBlock) Blocks.FIRE;
        Field postField = WallBlock.class.getDeclaredField("TEST_SHAPE_POST");
        postField.setAccessible(true);
        VoxelShape postShape = (VoxelShape) postField.get(null);
        Field wallsField = WallBlock.class.getDeclaredField("TEST_SHAPES_WALL");
        wallsField.setAccessible(true);
        @SuppressWarnings("unchecked")
        Map<Direction, VoxelShape> wallShapes =
                (Map<Direction, VoxelShape>) wallsField.get(null);
        for (Block block : BuiltInRegistries.BLOCK) {
            for (BlockState state : block.getStateDefinition().getPossibleStates()) {
                int fullMask = 0;
                int centerMask = 0;
                int rigidMask = 0;
                for (Direction direction : Direction.values()) {
                    if (state.isFaceSturdy(
                            EmptyBlockGetter.INSTANCE,
                            BlockPos.ZERO,
                            direction)) {
                        fullMask |= directionMask(direction);
                    }
                    if (state.isFaceSturdy(
                            EmptyBlockGetter.INSTANCE,
                            BlockPos.ZERO,
                            direction,
                            SupportType.CENTER)) {
                        centerMask |= directionMask(direction);
                    }
                    if (state.isFaceSturdy(
                            EmptyBlockGetter.INSTANCE,
                            BlockPos.ZERO,
                            direction,
                            SupportType.RIGID)) {
                        rigidMask |= directionMask(direction);
                    }
                }
                VoxelShape downFace = state.getCollisionShape(
                        EmptyBlockGetter.INSTANCE, BlockPos.ZERO).getFaceShape(Direction.DOWN);
                int wallCover = isCovered(postShape, downFace) ? 1 : 0;
                for (Direction direction : Direction.Plane.HORIZONTAL) {
                    if (isCovered(wallShapes.get(direction), downFace)) {
                        wallCover |= switch (direction) {
                            case NORTH -> 2;
                            case EAST -> 4;
                            case SOUTH -> 8;
                            case WEST -> 16;
                            default -> 0;
                        };
                    }
                }
                boolean flammable = (boolean) canBurn.invoke(fire, state);
                System.out.println(
                        Block.getId(state)
                                + "\t" + fullMask
                                + "\t" + Block.isExceptionForConnection(state)
                                + "\t" + flammable
                                + "\t" + centerMask
                                + "\t" + rigidMask
                                + "\t" + state.isSignalSource()
                                + "\t" + wallCover);
            }
        }
    }

    private static int directionMask(Direction direction) {
        return switch (direction) {
                            case DOWN -> 1;
                            case UP -> 2;
                            case NORTH -> 4;
                            case EAST -> 8;
                            case SOUTH -> 16;
                            case WEST -> 32;
                        };
    }

    private static boolean isCovered(VoxelShape test, VoxelShape actual) {
        return !Shapes.joinIsNotEmpty(test, actual, BooleanOp.ONLY_FIRST);
    }
}
