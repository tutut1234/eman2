find_package(OpenGL REQUIRED)

message_var(OPENGL_INCLUDE_DIR)
message_var(OPENGL_LIBRARIES)

if(OpenGL_FOUND)
	set_target_properties(OpenGL::GL PROPERTIES
						  INTERFACE_COMPILE_DEFINITIONS USE_OPENGL
						  )
		
endif()
